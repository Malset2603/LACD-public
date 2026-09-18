# 이 파일은 law articles list (law_list.csv) 에서 법률 링크를 받아 법률 전체 조문을 내려받는 코드이다.
def articles_extract(laws_dir = "./data/database/laws_list.csv"):

    import pandas as pd
    import xml.etree.ElementTree as ET
    from urllib.request import urlopen
    from tqdm import trange

    #앞서 본문 리스르 뽑아온 csv파일을 불러오는 단계
    law_list = pd.read_csv(laws_dir, encoding="UTF8")

    #조문단위에서 뽑아올 XML요소 정리
    contents = ['조문번호','조문가지번호','조문여부','조문제목','조문시행일자',
                '조문이동이전','조문이동이후','조문변경여부','조문제개정유형',
                '조문내용','조문참고자료','항']

    detail_dict = {}
    depth1_dict = {}
    depth2_dict = {}
    depth3_dict = {}

    # def remove_tag(content):
    #     cleaned_text = re.sub('<.*?>', '', content)
    #     return cleaned_text

    for i in trange(len(law_list)):
        #각 법령에 접근하기 위해 URL을 지정해주는 부분
        url_head = "https://www.law.go.kr/"
        detail_link = url_head + law_list.loc[i]['법령상세링크'].replace('HTML', 'XML')
        detail = urlopen(detail_link).read() # type: ignore
        root = ET.fromstring(detail)

        detail_dict = {}
        depth1_dict = {}
        depth2_dict = {}
        depth3_dict = {}

        detail_list = []


        for n in range(len(root[1])):
            # print(root[1][n].tag)
            for content in contents:
                dict_key = content
                
                try:
                    if(content == '항'):
                        if(root[1][n].find('항') is not None):
                            detail_dict['항'] = []
                            for depth1 in root[1][n].iter('항'):
                                if(depth1.find('호') is not None):
                                    depth1_dict['호'] = []
                                    for depth2 in depth1.iter('호'):
                                        if(depth2.find('목') is not None):
                                            depth2_dict['목'] = []
                                            for depth3 in depth2.iter('목'):
                                                if(depth3.find('목내용')is not None):
                                                    # print(depth3.find('목내용').text.strip())
                                                    depth3_dict['목내용'] = depth3.find('목내용').text.strip() # type: ignore
                                                    depth2_dict['목'].append(depth3_dict)
                                                    depth3_dict = {}
                                        if(depth2.find('호내용')is not None):
                                            # print(depth2.find('호내용').text.strip())
                                            depth2_dict['호내용'] = depth2.find('호내용').text.strip() # type: ignore
                                            depth1_dict['호'].append(depth2_dict)
                                            depth2_dict = {}
                                        else:
                                            depth1_dict['호'].append(depth2_dict)
                                            depth2_dict = {}
                                if(depth1.find('항내용') is not None):
                                    # print(depth1.find('항내용').text.strip())
                                    depth1_dict['항내용'] = depth1.find('항내용').text.strip() # type: ignore
                                    detail_dict['항'].append(depth1_dict)
                                    depth1_dict = {}
                                else:
                                    detail_dict['항'].append(depth1_dict)
                                    depth1_dict = {}
                        
                        
                    else:
                        dict_value = root[1][n].find(content).text.strip().replace('\n', '') # type: ignore
                        detail_dict[dict_key] = dict_value
                except:
                    # detail_dict[dict_key] = ''
                    continue
                
            detail_list.append(detail_dict)
            detail_dict = {}
        
        #파일이름을 법령명으로 하기 위한 부분
        file_name = law_list.loc[i]['법령명한글']
        if len(file_name)>32:
            file_name = law_list.loc[i]['법령약칭명']    
        path_format ='./data/database/laws/{0}.csv'.format(file_name)
        
        each_law_content = pd.DataFrame(detail_list)
        each_law_content.to_csv(path_format, index=False)
        
        detail_list = []

if __name__ == "__main__":
    articles_extract()

