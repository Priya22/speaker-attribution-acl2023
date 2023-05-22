import sys
sys.path.append('/h/vkpriya/bookNLP/booknlp-en')

from booknlpen.english.speaker_attribution import BERTSpeakerID
import torch.nn as nn
import torch
import argparse
import json, re, sys, string
from collections import Counter, defaultdict
import csv
import os
from random import shuffle
import sys
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_speaker_data(filename):

	with open(filename) as file:
		
		data={}

		for line in file:
			cols=line.rstrip().split("\t")
			
			sid=cols[0]
			qid = cols[1]
			eid=cols[2]
			cands=json.loads(cols[4])
			quote=int(cols[3]) #quote token position
			text=cols[5].split(" ")

			for s,e,_,_ in cands:
				#start and end token index in context
				if s > len(text) or e > len(text):
					print("reading problem", s, e, len(text))
					sys.exit(1)

			if sid not in data:
				data[sid]=[]
			data[sid].append((qid, eid, cands, quote, text))

		x=[]
		m=[]
		o=[]

		sids=list(data.keys())

		shuffle(sids)

		for sid in sids:
			for qid, eid, cands, quote, text in data[sid]:
				x.append(text)
				m.append((eid, cands, quote))
				o.append((sid, qid))


		return x, m, o

def predict_b(model, test_x_batches, test_m_batches, test_y_batches, test_o_batches, test_i_batches):
	model.eval()
	gold_eids = []
	pred_eids = []
	meta_info = []
	pred_confs = []

	with torch.no_grad():
		idd = 0
		for x1, m1, y1, o1, i1 in zip(test_x_batches, test_m_batches, test_y_batches, test_o_batches, test_i_batches):
			y_pred = model.forward(x1, m1)
			#SUM PREDS FOR EACH CANDIDTE?????????????

			predictions=y_pred.detach().cpu().numpy()
			orig, meta = o1
			for idx, preds in enumerate(predictions):
				# prediction = pred[0]
				ent2probs = {}
				ents = y1['eid'][idx]
				for e, p in zip(ents, preds):
					if e is None:
						continue
					if e not in ent2probs:
						ent2probs[e] = 0
					ent2probs[e] += p[0]
					
				if len(ent2probs) == 0:
					predval="none-%s"%(idd)
					predconf = 0.
				else:
					ent_probs = [(x,y) for x,y in ent2probs.items()]
					ent_probs = sorted(ent_probs, key=lambda x: x[1], reverse=True)
					predval = ent_probs[0][0]
					predconf = ent_probs[0][1]
#                 sent=orig[idx]
				# pred_conf = y_pred[idx][prediction].detach().cpu().numpy()
				
				# if prediction >= len(meta[idx][1]):
				# 	prediction=torch.argmax(y_pred[idx][:len(meta[idx][1])])
				# 	pred_conf = y_pred[idx][prediction].detach().cpu().numpy()

				gold_eids.append(y1["quote_eids"][idx])

				# predval=y1["eid"][idx][prediction]
				# if predval is None:
				# 	predval="none-%s" % (idd)
				pred_eids.append(predval)
				pred_confs.append(predconf)

				meta_info.append(i1[idx])
				idd += 1
				
	return gold_eids, pred_eids, pred_confs, meta_info

def predict(model, test_x_batches, test_m_batches, test_y_batches, test_o_batches, test_i_batches):
	model.eval()
	gold_eids = []
	pred_eids = []
	meta_info = []
	pred_confs = []

	with torch.no_grad():
		idd = 0
		for x1, m1, y1, o1, i1 in zip(test_x_batches, test_m_batches, test_y_batches, test_o_batches, test_i_batches):
			y_pred = model.forward(x1, m1)
			predictions=torch.argmax(y_pred, axis=1).detach().cpu().numpy()
			orig, meta = o1
			for idx, pred in enumerate(predictions):
				prediction = pred[0]

#                 sent=orig[idx]
				pred_conf = y_pred[idx][prediction].detach().cpu().numpy()[0]
				
				if prediction >= len(meta[idx][1]):
					prediction=torch.argmax(y_pred[idx][:len(meta[idx][1])])
					pred_conf = y_pred[idx][prediction].detach().cpu().numpy()[0]

				gold_eids.append(y1["quote_eids"][idx])

				predval=y1["eid"][idx][prediction]
				if predval is None:
					predval="none-%s" % (idd)
				pred_eids.append(predval)
				pred_confs.append(pred_conf)

				meta_info.append(i1[idx])
				idd += 1
				
	return gold_eids, pred_eids, pred_confs, meta_info

def resolve_sequential(pred_eids, meta_info, assigned_speakers):
	assigned_speakers = assigned_speakers.copy()

	novel2pinfo = {}
	for meta, pred in zip(meta_info, pred_eids):
		# idd = "_".join(meta)
		novel, qid = meta
		if novel not in novel2pinfo:
			novel2pinfo[novel] = []
		novel2pinfo[novel].append((qid, pred))
	
	resolved_pinfos = {}

	for novel, pinfos in novel2pinfo.items():
		qdf = pd.read_csv(os.path.join('/h/vkpriya/quoteAttr/data', novel, 'quote_info.csv'))
		qdf.sort_values(by='startByte', inplace=True)
		sorted_qids = {x:i for i,x in enumerate(qdf['qID'].tolist())}
		pinfos = sorted(pinfos, key=lambda x: sorted_qids[x[0]])
		res_preds = []
		for qid, pred in pinfos:
			idd = "_".join([novel, qid])
			if pred[0] == 'Q':
				ref_idd = "_".join([novel, pred])
				if ref_idd not in assigned_speakers:
					resolved_pinfos[idd] = "none-%s" % (idd)
				else:
					res_ent = assigned_speakers[ref_idd]
					resolved_pinfos[idd] = res_ent
					assigned_speakers[idd] = res_ent
			else:
				resolved_pinfos[idd] = pred
				assigned_speakers[idd] = pred
		
	new_pred_eids = []
	for novel, qid in meta_info:
		idd = "_".join([novel, qid])
		new_pred_eids.append(resolved_pinfos[idd])
	
	return new_pred_eids


if __name__ == "__main__":
	

	parser = argparse.ArgumentParser()
	parser.add_argument('--trainData', help='Filename containing training data', required=False)
	parser.add_argument('--devData', help='Filename containing dev data', required=False)
	parser.add_argument('--testData', help='Filename containing test data', required=False)
	parser.add_argument('--base_model', help='Base BERT model', required=False)
	parser.add_argument('--savePath', help='Folder to save outputs', required=False)

	args = vars(parser.parse_args())

	trainData=args["trainData"]
	devData=args["devData"]
	testData=args["testData"]
	base_model=args["base_model"]
	savePath=args["savePath"]

	os.makedirs(savePath, exist_ok=True)

	model_name = os.path.join(savePath, 'best_model.model')
	
	assigned_speakers = {}


	train_x, train_m, train_i=read_speaker_data(trainData)
	dev_x, dev_m, dev_i=read_speaker_data(devData)
	test_x, test_m, test_i=read_speaker_data(testData)

	#populate assigned speakers from training data
	for tm, ti in zip(train_m, train_i):
		gold = tm[0]
		idd = "_".join(ti)
		assigned_speakers[idd] = gold


	metric="accuracy"
	bertSpeaker=BERTSpeakerID(base_model=base_model)
	bertSpeaker.to(device)
	train_x_batches, train_m_batches, train_y_batches, train_o_batches, train_i_batches=bertSpeaker.get_batches(train_x, train_m, train_i)
	dev_x_batches, dev_m_batches, dev_y_batches, dev_o_batches, dev_i_batches=bertSpeaker.get_batches(dev_x, dev_m, dev_i)
	test_x_batches, test_m_batches, test_y_batches, test_o_batches, test_i_batches=bertSpeaker.get_batches(test_x, test_m, test_i)
	
	optimizer = torch.optim.Adam(bertSpeaker.parameters(), lr=1e-5)
	cross_entropy=nn.CrossEntropyLoss()

	best_dev_acc = 0.

	num_epochs=10

	for epoch in range(num_epochs):

		bertSpeaker.train()
		bigLoss=0

		for x1, m1, y1 in zip(train_x_batches, train_m_batches, train_y_batches):
			y_pred = bertSpeaker.forward(x1, m1)

			batch_y=y1["y"].unsqueeze(-1)
			batch_y=torch.abs(batch_y-1)*-100
			#NICE  #all true candidates are accepted
			true_preds=y_pred+batch_y

			golds_sum=torch.logsumexp(true_preds, 1)
			all_sum=torch.logsumexp(y_pred, 1)

			loss=torch.sum(all_sum-golds_sum)
			bigLoss+=loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print("\t\t\tEpoch %s loss: %.3f" % (epoch, bigLoss))
		
		# Evaluate; save the model that performs best on the dev data
		dev_F1, dev_acc=bertSpeaker.evaluate(dev_x_batches, dev_m_batches, dev_y_batches, dev_o_batches, epoch)
		sys.stdout.flush()
		if epoch % 1 == 0:
			if dev_F1 > best_dev_acc:
				torch.save(bertSpeaker.state_dict(), model_name)
				best_dev_acc = dev_F1
		
	# Test with best performing model on dev
	bertSpeaker.load_state_dict(torch.load(model_name, map_location=device))
	bertSpeaker.eval()

	test_F1, test_acc=bertSpeaker.evaluate(test_x_batches, test_m_batches, test_y_batches, test_o_batches, "test")
	print("Test F1:\t%.3f\t, accuracy:\t%.3f" % (test_F1, test_acc))

	gold_eids, pred_eids, pred_confs, meta_info = predict(bertSpeaker, dev_x_batches, dev_m_batches, dev_y_batches, \
										 dev_o_batches, dev_i_batches)
	
	resolved_pred_eids = resolve_sequential(pred_eids, meta_info, assigned_speakers)

	with open(os.path.join(savePath, 'val_preds_max.csv'), 'w') as f:
		writer = csv.writer(f)
		for meta, gold, pred, conf in zip(meta_info, gold_eids, resolved_pred_eids, pred_confs):
			writer.writerow([meta[0], meta[1], gold, pred, conf])

	
	gold_eids, pred_eids, pred_confs, meta_info = predict(bertSpeaker, test_x_batches, test_m_batches, test_y_batches, \
										 test_o_batches, test_i_batches)
	resolved_pred_eids = resolve_sequential(pred_eids, meta_info, assigned_speakers)

	with open(os.path.join(savePath, 'test_preds_max.csv'), 'w') as f:
		writer = csv.writer(f)
		for meta, gold, pred, conf in zip(meta_info, gold_eids, resolved_pred_eids, pred_confs):
			writer.writerow([meta[0], meta[1], gold, pred, conf])


	gold_eids, pred_eids, pred_confs, meta_info = predict_b(bertSpeaker, dev_x_batches, dev_m_batches, dev_y_batches, \
										 dev_o_batches, dev_i_batches)
	
	resolved_pred_eids = resolve_sequential(pred_eids, meta_info, assigned_speakers)

	with open(os.path.join(savePath, 'val_preds.csv'), 'w') as f:
		writer = csv.writer(f)
		for meta, gold, pred, conf in zip(meta_info, gold_eids, resolved_pred_eids, pred_confs):
			writer.writerow([meta[0], meta[1], gold, pred, conf])

	
	gold_eids, pred_eids, pred_confs, meta_info = predict_b(bertSpeaker, test_x_batches, test_m_batches, test_y_batches, \
										 test_o_batches, test_i_batches)
	resolved_pred_eids = resolve_sequential(pred_eids, meta_info, assigned_speakers)

	with open(os.path.join(savePath, 'test_preds.csv'), 'w') as f:
		writer = csv.writer(f)
		for meta, gold, pred, conf in zip(meta_info, gold_eids, resolved_pred_eids, pred_confs):
			writer.writerow([meta[0], meta[1], gold, pred, conf])

































































































































