# utilities for preprocessing


# 법률인지?
def is_law(law_list, source_law):
    if source_law in law_list:
        return True
    else:
        return False
    

# 법률 리스트 생성
# base execute dir = lawDB-LLM/
def generate_law_list(path='./data/database/'):
    law_list = []
    import csv
    with open(path+'laws_list.csv', mode='r', encoding='utf-8') as file:
        # csv reader 객체를 생성합니다.
        reader = csv.reader(file)
        
        # 각 행을 리스트에 추가합니다.
        for row in reader:
            # print(row)
            # row[10] == 법령구분명
            if row[10] =='법률' or row[10] =='헌법':
                # 법령명 한글
                law_list.append(row[2])
    
    return law_list

def generate_law_id_pair_list(path='./data/database/'):
    law_list = []
    import csv
    with open(path+'laws_list.csv', mode='r', encoding='utf-8') as file:
        # csv reader 객체를 생성합니다.
        reader = csv.reader(file)
        
        # 각 행을 리스트에 추가합니다.
        for row in reader:
            # print(row)
            # row[10] == 법령구분명
            if row[10] =='법률' or row[10] =='헌법':
                # 법령명 한글
                law_list.append((row[2], row[4]))
    
    return law_list

def recursive_data_extract(data):

    if isinstance(data, dict):
        ext = ""
        for v in data.values():
            if isinstance(v, str):
                ext = recursive_data_extract(v)+ext
            else:
                ext = ext+recursive_data_extract(v)
        return ext
    elif isinstance(data, list):
        if len(data) == 1:
            return recursive_data_extract(data[0])
        else:
            return recursive_data_extract(data[0])+recursive_data_extract(data[1:])
    # str 인 경우
    elif isinstance(data, str):
        return data
    else:
        return ""



def laws_name_list(laws_dir = "./data/database/laws_list.csv"):

    import pandas as pd    
    # reading CSV file
    data = pd.read_csv(laws_dir)
    
    # converting column data to list
    laws_name_list = data['법령명한글'].tolist()
    return laws_name_list


conflict_idx = 0

def what_law_exists(laws_name_list, target_text):
    global conflict_idx
    return_law = ""
    target_text = target_text.replace(" ", "")
    for law in laws_name_list:
        l = law.replace(" ","")
        # 서로 다른 두 개가 나오면 어떻게 할지 대책 세워야함.
        if l in target_text and ((return_law in law) or return_law == ""):
            return_law = law

        elif law in return_law:
            continue

        elif l in target_text and return_law != "":
            
            law_match = target_text.index(l)
            prev_law_match = target_text.index(return_law.replace(" ", ""))

            # print("conflict!, text:", target_text)
            # print(law, return_law)

            if prev_law_match < law_match:
                return_law = law

            # print("final return: {}".format(return_law))


        else:
            continue

    if return_law == "":
        print("conflict {}: ".format(conflict_idx)+target_text)
        conflict_idx = conflict_idx+1
    
    return return_law



# lawlist 와 한글 법령명을 받아서 lawid 반환
def lawname_to_lawid(law_list, lawname):
    for l in law_list:
        if l[0] == lawname:
            return l[1]
        
    return False

# article 을 받아서 article id 로 돌려줌
# e.g. 제13조->"13"
# e.g. 제45조의2->"45-2"
def articlename_to_article_id(articlename):
    articlename = articlename[1:]
    if "의" in articlename:
        articlename = articlename.replace("조의", "-")
    else:
        articlename = articlename.replace("조", "")
        
    return articlename