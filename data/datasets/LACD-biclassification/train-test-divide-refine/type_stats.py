import json
from collections import defaultdict
from src.methods.ReX.hybrid_result import classify_type
from src.methods.LawGNN.article_network.article_network import ArticleNetwork
from src.utils.utils import article_key_function
def analyze_type_stats():
    # ArticleNetwork 초기화
    article_network = ArticleNetwork()

    # 타입별 카운트를 저장할 딕셔너리들
    correct_type_counts = defaultdict(int)
    incorrect_type_counts = defaultdict(int) 
    total_type_counts = defaultdict(int)

    # train, val, test.jsonl 파일 읽기
    file_paths = [
        "./data/datasets/LACD-biclassification/train-test-divide-refine/train.jsonl",
        "./data/datasets/LACD-biclassification/train-test-divide-refine/val.jsonl",
        "./data/datasets/LACD-biclassification/train-test-divide-refine/test.jsonl"
    ]
    
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                article1 = article_key_function(data["article1"])
                article2 = article_key_function(data["article2"]) 
                answer = data["answer"]
                # 타입 분류
                pair_type = classify_type(article1, article2, article_network)
                
                # 전체 카운트 증가
                total_type_counts[pair_type] += 1
                
                # 정답/오답 카운트 증가
                if answer:
                    correct_type_counts[pair_type] += 1
                else:
                    incorrect_type_counts[pair_type] += 1

    # 결과 출력
    print("\n=== 타입별 통계 ===")
    print("\n타입별 정답 수:")
    for t, count in correct_type_counts.items():
        print(f"{t}: {count}")
        
    print("\n타입별 오답 수:")
    for t, count in incorrect_type_counts.items():
        print(f"{t}: {count}")
        
    print("\n타입별 전체 수:")
    for t, count in total_type_counts.items():
        print(f"{t}: {count}")
        
    # 타입별 정확도 계산 및 출력
    print("\n타입별 정확도:")
    for t in total_type_counts.keys():
        accuracy = correct_type_counts[t] / total_type_counts[t] * 100 if total_type_counts[t] > 0 else 0
        print(f"{t}: {accuracy:.2f}%")

if __name__ == "__main__":
    analyze_type_stats()
