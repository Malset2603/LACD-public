

# You should change my id to your id
def get_law_list(myid = None , path ='./data/database/'):

    import pandas as pd
    import xml.etree.ElementTree as ET
    from urllib.request import urlopen
    from tqdm import trange
    

    # xml url 불러오기
    url = f"http://www.law.go.kr/DRF/lawSearch.do?OC={myid}&target=law&type=XML"
    # urlopen을 활용하여 url 불러오기
    response = urlopen(url).read()
    # ET.fromstring을 통해 전체 불러오기
    xtree = ET.fromstring(response)
    # 전체 법령 개수 불러오기
    totalCnt = int(xtree.find('totalCnt').text)

    page = 1
    rows = []

    for i in trange(int(totalCnt / 20)+1):

        for node in xtree:
            try:
                법령일련번호 = node.find('법령일련번호').text
                현행연혁코드 = node.find('현행연혁코드').text
                법령명한글 = node.find('법령명한글').text
                법령약칭명 = node.find('법령약칭명').text
                법령ID = node.find('법령ID').text
                공포일자 = node.find('공포일자').text
                공포번호 = node.find('공포번호').text
                제개정구분명 = node.find('제개정구분명').text
                소관부처코드 = node.find('소관부처코드').text
                소관부처명 = node.find('소관부처명').text
                법령구분명 = node.find('법령구분명').text
                소관부처명 = node.find('소관부처명').text
                시행일자 = node.find('시행일자').text
                자법타법여부 = node.find('자법타법여부').text
                법령상세링크 = node.find('법령상세링크').text

                rows.append({'법령일련번호': 법령일련번호,
                            '현행연혁코드': 현행연혁코드,
                            '법령명한글': 법령명한글,
                            '법령약칭명': 법령약칭명,
                            '법령ID': 법령ID,
                            '공포일자': 공포일자,
                            '공포번호': 공포번호,
                            '제개정구분명': 제개정구분명,
                            '소관부처코드': 소관부처코드,
                            '소관부처명': 소관부처명,
                            '소관부처코드': 소관부처코드,
                            '법령구분명': 법령구분명,
                            '시행일자': 시행일자,
                            '자법타법여부': 자법타법여부,
                            '법령상세링크': 법령상세링크})
                
            except Exception as e:
                continue
        

        page += 1
        url = "http://www.law.go.kr/DRF/lawSearch.do?OC={}&target=law&type=XML&page={}".format(myid,page)
        response = urlopen(url).read()
        xtree = ET.fromstring(response)
        
    cases = pd.DataFrame(rows)
    cases.to_csv(path + 'laws_list.csv', index=False)

if __name__ == "__main__":
    get_law_list()