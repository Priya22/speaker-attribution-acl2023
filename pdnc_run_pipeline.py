from booknlpen.english.english_booknlp import EnglishBookNLP as BookNLP 
import os

model_params={
		"pipeline":"entity,quote,supersense,event,coref", 
		"model":"big"
	}
	
booknlp = BookNLP(model_params)

IN_ROOT = 'data/pdnc_source'
OUT_ROOT = 'booknlpen/pdnc_output'
os.makedirs(OUT_ROOT, exist_ok=True)

novels = []
ip_paths = []
op_paths = []

for novel in os.scandir(IN_ROOT):
    if os.path.isdir(novel) and novel.name[0] not in ['.', '_']:
        novel_name = novel.name
        novels.append(novel_name)
        ip_paths.append(os.path.join(novel.path, 'novel.txt'))
        op_path = os.path.join(OUT_ROOT, novel_name)
        if not os.path.isdir(op_path):
            os.mkdir(op_path)

        op_paths.append(op_path)

assert len(novels) == len(ip_paths) == len(op_paths), print("Length Mismatch!")

for bname, ip, op in zip(novels, ip_paths, op_paths):
    print("Processing: ", bname)
    booknlp.process(ip, op, bname)
    print("Done")