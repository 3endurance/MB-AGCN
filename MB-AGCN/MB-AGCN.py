import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from DataHandler import negSamp, transpose, DataHandler, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle

class Recommender:
	def __init__(self, sess, handler):
		self.sess = sess
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'HR', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		for ep in range(stloc, args.epoch):
			test = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % args.tstEpoch == 0:
				self.saveHistory()
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		self.saveHistory()
		# 消息传播激活函数是relu

	def messagePropagate(self, lats, adj, lats2):

		# a = tf.sparse.sparse_dense_matmul(adj, lats)
		# a = a + tf.sparse.sparse_dense_matmul(adj, lats)
		# return Activate(a, self.actFunc)
		return Activate(tf.sparse_tensor_dense_matmul(adj, lats), self.actFunc)

	def specialize(self, uEmbed, iEmbed, params):
		retUEmbed = tf.reduce_sum(tf.expand_dims(uEmbed, axis=-1) * params['uW1'], axis=1)
		retUEmbed = tf.reduce_sum(tf.expand_dims(retUEmbed, axis=-1) * params['uW2'], axis=1)
		retUEmbed = tf.concat([retUEmbed, uEmbed], axis=-1)
		retIEmbed = tf.reduce_sum(tf.expand_dims(iEmbed, axis=-1) * params['iW1'], axis=1)
		retIEmbed = tf.reduce_sum(tf.expand_dims(retIEmbed, axis=-1) * params['iW2'], axis=1)
		retIEmbed = tf.concat([retIEmbed, iEmbed], axis=-1)
		return retUEmbed, retIEmbed

	# 定义model

	def defineModel(self):
		# 用户商品嵌入向量

		uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim//2], reg=True)
		iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim//2], reg=True)
		behEmbeds = NNs.defineParam('behEmbeds', [args.behNum + 1, args.latdim//2])
		self.ulat = [0] * (args.behNum + 1)
		self.ilat = [0] * (args.behNum + 1)
		# 不同的行为
		for beh in range(args.behNum):
			params = self.metaForSpecialize(uEmbed0, iEmbed0, behEmbeds[beh], [self.adjs[beh]], [self.tpAdjs[beh]])
			behUEmbed0, behIEmbed0 = self.specialize(uEmbed0, iEmbed0, params)
			# behUEmbed0 = uEmbed0
			# behIEmbed0 = iEmbed0
			ulats = [behUEmbed0]
			ilats = [behIEmbed0]
			for i in range(args.gnn_layer):
				ulat = self.messagePropagate(ilats[-1], self.adjs[beh], ulats[-1])
				ilat = self.messagePropagate(ulats[-1], self.tpAdjs[beh], ilats[-1])
				# score3 = np.dot(ulat,ulats)
				# score4 = np.dot(ilat,ilats)
				ulats.append(ulat + ulats[-1])
				ilats.append(ilat + ilats[-1])
			self.ulat[beh] = tf.add_n(ulats)
			self.ilat[beh] = tf.add_n(ilats)

		params = self.metaForSpecialize(uEmbed0, iEmbed0, behEmbeds[-1], self.adjs, self.tpAdjs)
		behUEmbed0, behIEmbed0 = self.specialize(uEmbed0, iEmbed0, params)
		ulats = [behUEmbed0]
		ilats = [behIEmbed0]
		# 多层传播得到用户和项目的最终嵌入表示
		for i in range(args.gnn_layer):

			ubehLats = []
			ibehLats = []
			for beh in range(args.behNum):
				ulat = self.messagePropagate(ilats[-1], self.adjs[beh], ulats[-1])
				ilat = self.messagePropagate(ulats[-1], self.tpAdjs[beh], ilats[-1])
				# ulats.append(ulat + ulats[-1])
				# ilats.append(ilat + ilats[-1])
				ubehLats.append(ulat)
				ibehLats.append(ilat)
			ulat = tf.add_n(NNs.lightSelfAttention(ubehLats, args.behNum, args.latdim, args.att_head))
			ilat = tf.add_n(NNs.lightSelfAttention(ibehLats, args.behNum, args.latdim, args.att_head))
			ulats.append(ulat)
			ilats.append(ilat)
		self.ulat[-1] = tf.add_n(ulats)
		self.ilat[-1] = tf.add_n(ilats)


	def _predict(self, ulat, ilat, params):
		# predEmbed = tf.expand_dims(tf.concat([ulat * ilat, ulat, ilat], axis=-1), axis=1)

		predEmbed = tf.expand_dims(tf.concat([ulat * ilat,ulat,ilat], axis=-1), axis=1)
		predEmbed = Activate(predEmbed @ params['w1'] + params['b1'], self.actFunc)
		preds = tf.squeeze(predEmbed @ params['w2'])
		return preds

	def predict(self, src, tgt):
		uids = self.uids[tgt]
		iids = self.iids[tgt]
		src_ulat = tf.nn.embedding_lookup(self.ulat[src], uids)
		src_ilat = tf.nn.embedding_lookup(self.ilat[src], iids)
		tgt_ulat = tf.nn.embedding_lookup(self.ulat[tgt], uids)
		tgt_ilat = tf.nn.embedding_lookup(self.ilat[tgt], iids)
		predParams = self.metaForPredict(src_ulat, src_ilat, tgt_ulat, tgt_ilat)
		return self._predict(src_ulat, src_ilat, predParams) * args.mult

	def prepareModel(self):
		# 激活函数
		self.actFunc = 'leakyRelu'
		# adjs为列表，存储着一阶和二阶的子图对应的邻接矩阵
		self.adjs = []
		# 邻接矩阵
		self.tpAdjs = []
		# 用户ID 商品ID
		self.uids, self.iids = [], []
		for i in range(args.behNum):

			adj = self.handler.trnMats[i]

			idx, data, shape = transToLsts(adj, norm=True)
			# self.adjs.append(tf.sparse.SparseTensor(idx, data, shape))
			self.adjs.append(tf.SparseTensor(idx, data, shape))

			idx, data, shape = transToLsts(transpose(adj), norm=True)
			self.tpAdjs.append(tf.SparseTensor(idx, data, shape))
			# tf.placeholder：用于得到传递进来的真实的训练样本：
			self.uids.append(tf.placeholder(name='uids'+str(i), dtype=tf.int32, shape=[None]))
			self.iids.append(tf.placeholder(name='iids'+str(i), dtype=tf.int32, shape=[None]))
		
		self.defineModel()
		self.preLoss = 0
		for src in range(args.behNum + 1):
			for tgt in range(args.behNum):
				preds = self.predict(src, tgt)
				sampNum = tf.shape(self.uids[tgt])[0] // 2
				posPred = tf.slice(preds, [0], [sampNum])
				negPred = tf.slice(preds, [sampNum], [-1])
				self.preLoss += tf.reduce_mean(tf.maximum(0.0, 1.0 - (posPred - negPred)))
				if src == args.behNum and tgt == args.behNum - 1:
					self.targetPreds = preds
		self.regLoss = args.reg * Regularize()
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def sampleTrainBatch(self, batIds, labelMat):

		temLabel = labelMat[batIds].toarray()
		batch = len(batIds)
		temlen = batch * 2 * args.sampNum

		uLocs = [None] * temlen
		iLocs = [None] * temlen
		cur = 0
		for i in range(batch):
			posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
			sampNum = min(args.sampNum, len(posset))
			if sampNum == 0:
				poslocs = [np.random.choice(args.item)]
				neglocs = [poslocs[0]]
			else:
				poslocs = np.random.choice(posset, sampNum)
				neglocs = negSamp(temLabel[i], sampNum, args.item)
			for j in range(sampNum):
				posloc = poslocs[j]
				negloc = neglocs[j]
				uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
				iLocs[cur] = posloc
				iLocs[cur+temlen//2] = negloc
				cur += 1
		uLocs = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
		iLocs = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
		return uLocs, iLocs

	def trainEpoch(self):
		num = args.user
		sfIds = np.random.permutation(num)[:args.trnNum]
		epochLoss, epochPreLoss = [0] * 2
		num = len(sfIds)
		steps = int(np.ceil(num / args.batch))

		for i in range(steps):
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batIds = sfIds[st: ed]

			target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
			feed_dict = {}
			for beh in range(args.behNum):
				uLocs, iLocs = self.sampleTrainBatch(batIds, self.handler.trnMats[beh])
				feed_dict[self.uids[beh]] = uLocs
				feed_dict[self.iids[beh]] = iLocs

			res = self.sess.run(target, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

			preLoss, regLoss, loss = res[1:]

			epochLoss += loss
			epochPreLoss += preLoss
			log('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		return ret

	def sampleTestBatch(self, batIds, labelMat):
		batch = len(batIds)
		temTst = self.handler.tstInt[batIds]
		temLabel = labelMat[batIds].toarray()
		temlen = batch * 100
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		tstLocs = [None] * batch
		cur = 0
		for i in range(batch):
			posloc = temTst[i]
			negset = np.reshape(np.argwhere(temLabel[i]==0), [-1])
			rdnNegSet = np.random.permutation(negset)[:99]
			locset = np.concatenate((rdnNegSet, np.array([posloc])))
			tstLocs[i] = locset
			for j in range(100):
				uLocs[cur] = batIds[i]
				iLocs[cur] = locset[j]
				cur += 1
		return uLocs, iLocs, temTst, tstLocs

	def testEpoch(self):
		epochHit, epochNdcg = [0] * 2
		ids = self.handler.tstUsrs
		num = len(ids)
		tstBat = args.batch
		steps = int(np.ceil(num / tstBat))
		for i in range(steps):
			st = i * tstBat
			ed = min((i+1) * tstBat, num)
			batIds = ids[st: ed]
			feed_dict = {}
			uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, self.handler.trnMats[-1])
			feed_dict[self.uids[-1]] = uLocs
			feed_dict[self.iids[-1]] = iLocs
			preds = self.sess.run(self.targetPreds, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			hit, ndcg = self.calcRes(np.reshape(preds, [ed-st, 100]), temTst, tstLocs)
			epochHit += hit
			epochNdcg += ndcg
			log('Steps %d/%d: hit = %d, ndcg = %d          ' % (i, steps, hit, ndcg), save=False, oneline=True)
		ret = dict()
		ret['HR'] = epochHit / num
		ret['NDCG'] = epochNdcg / num
		return ret

	# 计算结果hit 和 ndcg
	def calcRes(self, preds, temTst, tstLocs):
		hit = 0
		ndcg = 0
		for j in range(preds.shape[0]):
			predvals = list(zip(preds[j], tstLocs[j]))
			predvals.sort(key=lambda x: x[0], reverse=True)
			shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
			if temTst[j] in shoot:
				hit += 1
				ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
		return hit, ndcg
	
	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

		# 加载模型

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	

if __name__ == '__main__':
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	with tf.Session(config=config) as sess:
		recom = Recommender(sess, handler)
		recom.run()
