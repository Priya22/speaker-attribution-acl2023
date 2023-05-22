import os

def read_tagset(filename):
	tags={}
	with open(filename) as file:
		for line in file:
			cols=line.rstrip().split("\t")
			tags[cols[0]]=int(cols[1])
	return tags


def read_filenames(filename):
	inpaths=[]
	outpaths=[]
	with open(filename) as file:
		for line in file:
			cols=line.rstrip().split("\t")
			
			if len(cols) == 2:
				inpaths.append(cols[0])
				outpaths.append(cols[1])

	return inpaths, outpaths


def read_booknlp(path, model):

	sentences=[]
	original_sentences=[]

	sentence=[]
	sentence.append(["[CLS]"])
	
	orig_sentence=[]
	orig_sentence.append(("[CLS]", -1))
	length=0
	
	max_sentence_length=400


	with open(path, encoding="utf-8") as file:
		header=file.readline().split("\t")

		s_idx=header.index("sentenceID")
		t_idx=header.index("tokenId")
		w_idx=header.index("originalWord")
		
		lastSentence=None

		for line in file:
			cols=line.rstrip().split("\t")
			s_id=cols[s_idx]
			t_id=cols[t_idx]
			w=cols[w_idx]

			toks=model.tokenizer.tokenize(w)
			
			if s_id != lastSentence or length + len(toks) > max_sentence_length:
				if len(sentence) > 0:
					sentence.append(["[SEP]"])
					sentences.append(sentence)

					orig_sentence.append((-1, "[SEP]"))
					original_sentences.append(orig_sentence)

					sentence=[]
					sentence.append(["[CLS]"])

					orig_sentence=[]
					orig_sentence.append((-1, "[CLS]"))

					length=0

			length+=len(toks)

			sentence.append([w])
			orig_sentence.append((t_id, w))

			lastSentence=s_id

	if len(sentence) > 1:
		sentence.append(["[SEP]"])
		sentences.append(sentence)

		orig_sentence.append((-1, "[SEP]"))
		original_sentences.append(orig_sentence)

	return sentences, original_sentences


def read_annotations(filename, tagset, labeled, model, doLowerCase=False):

	""" Read tsv data and return sentences and [word, tag, sentenceID, filename] list """

	with open(filename, encoding="utf-8") as f:
		sentence = []
		sentence.append(["[CLS]", -100, -100, -100, -100, -100, -100 -1, -1, None])
		sentences = []
		sentenceID=0

		wp_tokens=0
		for line in f:
			if len(line) > 0:
				if line == '\n':
					sentenceID+=1

					sentence.append(["[SEP]", -100, -100, -100, -100, -100, -100 -1, -1, None])

					if wp_tokens > 510:
						print("too big! %s %s" % (wp_tokens, ' '.join([x[0] for x in sentence])))
						print()
						# sys.exit(1)

					if len(sentence) > 2 and wp_tokens < 510:
						sentences.append(sentence)
					
					wp_tokens=1

					sentence = []
					sentence.append(["[CLS]", -100, -100, -100, -100, -100, -100 -1, -1, None])


				else:
					data=[]
					split_line = line.rstrip().split('\t')

					word=split_line[0]

					if doLowerCase:
						if word[0].lower() != word[0]:
							word="[CAP] " + word.lower()
						else:
							word=word.lower()
						
					data.append(word)
					wp_tokens+=len(model.tokenizer.tokenize(word))

					if len(split_line) > 1 and labeled:
						data.append(tagset[split_line[1]] if labeled else 0)
					else:
						data.append(tagset["O"])

					if len(split_line) > 2 and labeled:
						data.append(tagset[split_line[2]] if labeled else 0)
					else:
						data.append(tagset["O"])

					if len(split_line) > 3 and labeled:
						data.append(tagset[split_line[3]] if labeled else 0)
					else:
						data.append(tagset["O"])


					# data.append(tagset[split_line[4]] if labeled else 0)

					if len(split_line) > 4 and labeled:
						data.append(tagset[split_line[4]] if labeled else 0)
					else:
						data.append(tagset["O"])

					if len(split_line) > 5 and labeled:
						data.append(tagset[split_line[5]] if labeled else 0)
					else:
						data.append(tagset["O"])

					data.append(sentenceID)
					data.append(filename)

					sentence.append(data)

		
		sentence.append(["[SEP]", -100, -100, -100, -100, -100, -100 -1, -1, None])
		if wp_tokens > 510:
			print("too big! %s" % (wp_tokens))
			# sys.exit(1)

		if len(sentence) > 2 and wp_tokens < 510:
			sentences.append(sentence)

	return sentences

def prepare_annotations_from_file(filename, tagset, model, labeled=True):

	""" Read a single file of annotations, returning:
		-- a list of sentences, each a list of [word, label, sentenceID, filename]
	"""

	sentences = []
	annotations = read_annotations(filename, tagset, labeled, model)
	sentences += annotations
	return sentences

def prepare_annotations_from_folder(folder, tagset, model, labeled=True, doLowerCase=False):

	""" Read folder of annotations, returning:
		-- a list of sentences, each a list of [word, label, sentenceID, filename]
	"""

	sentences = []
	for filename in os.listdir(folder):
		print(filename)
		annotations = read_annotations(os.path.join(folder,filename), tagset, labeled, model, doLowerCase=doLowerCase)
		sentences += annotations
	print("num sentences: %s" % len(sentences))
	return sentences
