from src.methods.ReX.hybrid_result import build_true_dict, build_true_dict_by_test, compute_ndcg, classify_type
import json
from tqdm import tqdm
import copy
import os
import numpy as np
from src.methods.LawGNN.article_network.article_network import ArticleNetwork
from tabulate import tabulate
import scipy.stats as stats

# ANSI escape codes
BLUE_BOLD = '\033[1;34m'
UNDERLINE = '\033[4m'
RESET = '\033[0m'

def highlight_table(table, header):
    # table: List[List[str or float]]
    # header: List[str]
    # 각 column별로 float값만 추출
    num_cols = len(header)
    col_values = [[] for _ in range(num_cols)]
    for row in table:
        for i, val in enumerate(row):
            try:
                # ±로 구분된 평균값만 추출
                if isinstance(val, str) and '±' in val:
                    col_values[i].append(float(val.split('±')[0]))
                else:
                    col_values[i].append(float(val))
            except:
                col_values[i].append(None)
    # 각 col별로 최고, 두번째 값 인덱스 찾기 (0번은 모델명)
    best_idx = [None] * num_cols
    second_idx = [None] * num_cols
    for j in range(1, num_cols):
        vals = [(i, v) for i, v in enumerate(col_values[j]) if v is not None]
        if not vals:
            continue
        vals_sorted = sorted(vals, key=lambda x: x[1], reverse=True)
        if len(vals_sorted) > 0:
            best_idx[j] = vals_sorted[0][0]
        if len(vals_sorted) > 1:
            second_idx[j] = vals_sorted[1][0]
    # 스타일 적용
    styled_table = []
    for i, row in enumerate(table):
        styled_row = []
        for j, val in enumerate(row):
            sval = str(val)
            if j == 0:
                styled_row.append(sval)
            elif best_idx[j] == i:
                styled_row.append(f"{BLUE_BOLD}{sval}{RESET}")
            elif second_idx[j] == i:
                styled_row.append(f"{UNDERLINE}{sval}{RESET}")
            else:
                styled_row.append(sval)
        styled_table.append(styled_row)
    return styled_table

def main(result_path, article_network):
    top_k = [5, 10, 50]
    top_k0 = [50]

    true_dicts = build_true_dict()
    # true_dicts = build_true_dict_by_test()

    # 결과 파일 읽기
    processed_query = set()

    results = []
    with open(result_path, 'r', encoding='utf-8') as result_file:
        for line in result_file:
            row = json.loads(line.strip())
            results.append(row)
            processed_query.add(row['article_to_check'])

    all_positives = 0
    for q in processed_query:
        all_positives += len(true_dicts[q])

    # metric 결과 저장용 딕셔너리
    table_metrics = {k: {"nDCG": 0.0, "Recall": 0.0, "Retrieval F1": 0.0} for k in top_k}

    for k0 in top_k0:
        results_as_binary_markings = []
        all_instances = 0

        # 각 인스턴스(쿼리)에 대해 articles를 rerank하고 binary marking 계산

        true_articles_count = 0
        for r in tqdm(results):
            query = r["article_to_check"]
            instance = {"article_to_check": query, "articles_as_binary": [], "articles_as_types": []}

            true_articles = set(true_dicts.get(query, []))
            true_articles_count += len(true_articles)
            for a in r["articles"]:
                instance["articles_as_binary"].append(1 if a in true_articles else 0)
                instance["articles_as_types"].append(
                    classify_type(query, a, article_network) if a in true_articles else "NONE"
                )
            all_instances += len(instance["articles_as_binary"])
            results_as_binary_markings.append(instance)

        print(f"True articles count: {true_articles_count}")
        print(f"Average true articles count: {true_articles_count / len(results)}")
        for k in top_k:
            total_true_positives = 0
            total_retrieved = 0
            recall = 0
            recall_cap = 0
            total_ndcg = 0.0
            possible_positives = 0

            recall = 0
            recall_cap = 0
            for r in results_as_binary_markings:
                binary_list = r["articles_as_binary"]
                topk_list = binary_list[:k]
                total_true_positives += sum(topk_list)
                total_retrieved += len(topk_list)
                recall += sum(topk_list) / len(true_dicts[r['article_to_check']])
                recall_cap += sum(topk_list) / min(k, len(true_dicts[r['article_to_check']]))
                total_ndcg += compute_ndcg(
                    binary_list,
                    [1 for i in range(len(true_dicts[r['article_to_check']]))] + [0 for j in range(k-len(true_dicts[r['article_to_check']]))],
                    k
                )

            recall = recall / len(results_as_binary_markings) * 100
            recall_cap = recall_cap / len(results_as_binary_markings) * 100
            precision = (total_true_positives / total_retrieved * 100) if total_retrieved > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            ndcg = total_ndcg / len(results_as_binary_markings) * 100

            # 표용 metric 저장
            table_metrics[k]["nDCG"] = ndcg
            table_metrics[k]["Recall"] = recall
            table_metrics[k]["Retrieval F1"] = f1

    # 모델명 추출
    model_name = os.path.basename(result_path).split(".")[0]
    row = [model_name]
    for metric in ["nDCG", "Recall", "Retrieval F1"]:
        for k in [5, 10, 50]:
            row.append(f"{table_metrics[k][metric]:.2f}")
    return row

def evaluate_fold(fold_dir, article_network):
    # 해당 폴드 디렉토리 내 모든 파일 경로를 리스트로 수집
    paths = [
        os.path.join(fold_dir, f)
        for f in os.listdir(fold_dir)
        if os.path.isfile(os.path.join(fold_dir, f))
    ]
    paths.sort()
    
    # 표 헤더
    header = ["Model"] + [f"nDCG@{k}" for k in [5,10,50]] + [f"Recall@{k}" for k in [5,10,50]] + [f"F1@{k}" for k in [5,10,50]]
    table = []
    
    for p in paths:
        try:
            print(f"처리 중: {p}")
            row = main(p, article_network)
            table.append(row)
        except Exception as e:
            print(f"오류 발생 {p}: {e}")
            continue
    
    return table, header

def aggregate_results(all_fold_results):
    # 모든 폴드의 결과를 모델별로 평균 계산
    model_results = {}
    
    for fold_name, table, _ in all_fold_results:
        for row in table:
            model_name = row[0]
            if model_name not in model_results:
                model_results[model_name] = [[] for _ in range(len(row)-1)]
            
            # 각 지표별 값 저장
            for i in range(1, len(row)):
                try:
                    model_results[model_name][i-1].append(float(row[i]))
                except:
                    pass
    
    # 평균과 표준편차 계산하여 최종 테이블 생성
    final_table = []
    for model_name, metrics in model_results.items():
        avg_row = [model_name]
        for metric_values in metrics:
            if metric_values:
                avg = np.mean(metric_values)
                std = np.std(metric_values)
                avg_row.append(f"{avg:.2f}±{std:.2f}")
            else:
                avg_row.append("N/A")
        final_table.append(avg_row)
    
    return final_table, model_results

def perform_statistical_tests(model_results, header):
    """모델 간 성능 차이의 통계적 유의성 검정"""
    # 모델 이름 목록
    model_names = list(model_results.keys())
    if len(model_names) < 2:
        print("통계 검정을 위한 모델이 충분하지 않습니다.")
        return
    
    # 각 지표별로 통계 검정 수행
    test_results = []
    
    for i in range(len(header) - 1):  # 첫 번째 열(모델명)은 제외
        metric_name = header[i + 1]
        print(f"\n{metric_name}에 대한 통계 검정:")
        
        # 모든 모델 쌍에 대해 t-test 수행
        for j in range(len(model_names)):
            for k in range(j + 1, len(model_names)):
                model1 = model_names[j]
                model2 = model_names[k]
                
                # 두 모델의 해당 지표 값 가져오기
                values1 = model_results[model1][i]
                values2 = model_results[model2][i]
                
                if len(values1) > 0 and len(values2) > 0:
                    # 두 모델이 동일한 폴드에 대해 평가되었으므로 쌍체 t-검정 수행
                    if len(values1) == len(values2):
                        t_stat, p_value = stats.ttest_rel(values1, values2)
                        test_type = "쌍체 t-검정"
                    else:
                        # 길이가 다른 경우 독립 표본 t-검정 수행
                        t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                        test_type = "독립 표본 t-검정"
                    
                    # 결과 저장 및 출력
                    significance = ""
                    if p_value < 0.001:
                        significance = "***"
                    elif p_value < 0.01:
                        significance = "**"
                    elif p_value < 0.05:
                        significance = "*"
                    
                    mean1 = np.mean(values1)
                    mean2 = np.mean(values2)
                    
                    result = {
                        "Metric": metric_name,
                        "Model1": model1,
                        "Model2": model2,
                        "Model1_Mean": mean1,
                        "Model2_Mean": mean2,
                        "t-statistic": t_stat,
                        "p-value": p_value,
                        "Significance": significance
                    }
                    
                    test_results.append(result)
                    
                    # 더 높은 성능을 보이는 모델 표시
                    better_model = model1 if mean1 > mean2 else model2
                    worse_model = model2 if better_model == model1 else model1
                    better_mean = max(mean1, mean2)
                    worse_mean = min(mean1, mean2)
                    
                    if p_value < 0.05:  # 유의수준 0.05
                        print(f"  {better_model}({better_mean:.2f}) > {worse_model}({worse_mean:.2f}): {test_type}, t={t_stat:.4f}, p={p_value:.4f} {significance}")
                    else:
                        print(f"  {model1}({mean1:.2f}) vs {model2}({mean2:.2f}): {test_type}, t={t_stat:.4f}, p={p_value:.4f}")
    
    return test_results

if __name__ == "__main__":
    import os
    base_dir = "./outputs/retrieval_results/multi-fold/"
    article_network = ArticleNetwork()
    
    # fold_0부터 fold_5까지 모든 폴드 디렉토리 찾기
    fold_dirs = []
    for i in range(6):  # 0부터 5까지
        fold_path = os.path.join(base_dir, f"fold_{i}")
        if os.path.isdir(fold_path):
            fold_dirs.append((f"fold_{i}", fold_path))
    
    # 각 폴드별 결과 저장
    all_fold_results = []
    
    for fold_name, fold_path in fold_dirs:
        print(f"\n{'='*50}")
        print(f"폴드 평가 중: {fold_name}")
        print(f"{'='*50}")
        
        table, header = evaluate_fold(fold_path, article_network)
        
        print(f"\n{fold_name} 결과:")
        print("-" * (len(header) * 15))
        styled_table = highlight_table(table, header)
        print(tabulate(styled_table, headers=header, tablefmt="github"))
        print("-" * (len(header) * 15))
        
        all_fold_results.append((fold_name, table, header))
    
    # 모든 폴드 결과 종합
    print(f"\n{'='*50}")
    print("모든 폴드 결과 종합")
    print(f"{'='*50}")
    
    final_table, model_results = aggregate_results(all_fold_results)
    styled_final_table = highlight_table(final_table, header)
    print(tabulate(styled_final_table, headers=header, tablefmt="github"))
    print("-" * (len(header) * 15))
    
    # 통계적 유의성 검정 수행
    print(f"\n{'='*50}")
    print("모델 간 통계적 유의성 검정")
    print(f"{'='*50}")
    test_results = perform_statistical_tests(model_results, header)
