







def eval_gutentag():
    
    major_chars = json.load(open('/h/vkpriya/bookNLP/booknlp-en/coref/novel2majorchars.json', 'r'))
    all_chars = json.load(open('/h/vkpriya/bookNLP/booknlp-en/coref/novel2allchars.json', 'r'))

    rows = []
    for novel in novels:
        cur_row = [novel]
        ent_df = pd.read_csv(os.path.join('/h/vkpriya/bookNLP/booknlp-en/booknlpen/pdnc_output', novel, novel+'.entities'), sep='\t', quoting=3, lineterminator='\n')
        ent_df = ent_df[(ent_df['cat']=='PER')&(ent_df['prop']=='PROP')]

        #major first
        name2id = {}
        id2names = {}
        for ind,row in enumerate(major_chars[novel]):
            id_ = row[0]
            for name in row[-1]:
                name2id[name] = id_
            id2names[id_] = row[-1]
        
        cur_row.append(len(id2names))
       
        ent_names = set(ent_df['text'].tolist())
        ent_matches = {}
        id_ent_matches = {}
        for ent in ent_names:
            if ent in name2id:
                id_ = name2id[ent]
                ent_matches[ent] = id_
                if id_ not in id_ent_matches:
                    id_ent_matches[id_] = []
                id_ent_matches[id_].append(ent)

        cur_row.append(len(id_ent_matches))

        #all
        name2id = {}
        id2names = {}
        for ind,row in enumerate(all_chars[novel]):
            id_ = row[0]
            for name in row[-1]:
                name2id[name] = id_
            id2names[id_] = row[-1]
        
        cur_row.append(len(id2names))
       
        ent_names = set(ent_df['text'].tolist())
        ent_matches = {}
        id_ent_matches = {}
        for ent in ent_names:
            if ent in name2id:
                id_ = name2id[ent]
                ent_matches[ent] = id_
                if id_ not in id_ent_matches:
                    id_ent_matches[id_] = []
                id_ent_matches[id_].append(ent)

        cur_row.append(len(id_ent_matches))

        rows.append(cur_row)
    
    rdf = pd.DataFrame(rows, columns=['novel', 'pdnc_num_major_ents', 'booknlp_num_major_matches', 'pdnc_num_ents', 'booknlp_num_matches'])

    return rdf