import os
import json
from collections import defaultdict

def calculate_statistics_and_probabilities(directory):
    files_data = []
    total_word_sum = 0
    total_pairs = 0
    all_answers_count = {'true': 0, 'false': 0}
    pairs_list = []  # collect all (a1, a2, answer) triples

    # 1) Read files, compute original stats, and build pairs_list
    for filename in os.listdir(directory):
        if not filename.endswith('.jsonl'):
            continue
        if filename == 'test.jsonl':
            continue
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            article_word_sum = 0
            article_count = 0
            answer_count = {'true': 0, 'false': 0}

            for line in f:
                data = json.loads(line)
                a1, a2, ans = data['article1'], data['article2'], data['answer']
                # count words
                article1_words = len(a1.split())
                article2_words = len(a2.split())
                article_word_sum += (article1_words + article2_words)
                article_count += 1

                # count answers
                if ans is True:
                    answer_count['true'] += 1
                else:
                    answer_count['false'] += 1

                # collect for probability computations
                pairs_list.append((a1, a2, ans))

            avg_word_sum = article_word_sum / article_count if article_count > 0 else 0
            files_data.append({
                'file': filename,
                'average_word_length_sum': avg_word_sum,
                'answer_count': answer_count,
            })

            total_word_sum += article_word_sum
            total_pairs += article_count
            all_answers_count['true'] += answer_count['true']
            all_answers_count['false'] += answer_count['false']

    overall_avg_word_length_sum = total_word_sum / total_pairs if total_pairs > 0 else 0

    # 2) Build helper structures for conditional probabilities
    #    - pairs_by_article1[a1] = list of (a1, a2, ans)
    #    - pair_map[(a1, a2)] = ans
    pairs_by_article1 = defaultdict(list)
    pair_map = {}
    for a1, a2, ans in pairs_list:
        pairs_by_article1[a1].append((a1, a2, ans))
        pair_map[(a1, a2)] = ans

    # 3) (1) P(answer=True)
    total_trues = sum(1 for _, _, ans in pairs_list if ans is True)
    prob_answer_true = total_trues / len(pairs_list) if pairs_list else 0

    # 4) (2) P(ans(a2,a3)=True | ans(a1,a2)=True)
    denom2 = 0
    num2 = 0
    for a1, a2, ans1 in pairs_list:
        if not ans1:
            continue
        for _, a3, ans2 in pairs_by_article1.get(a2, []):
            denom2 += 1
            if ans2:
                num2 += 1
    prob_cond_12_to_23 = num2 / denom2 if denom2 > 0 else 0

    # 5) (3) P(ans(a1,a3)=True | ans(a1,a2)=True and ans(a2,a3)=True)
    denom3 = 0
    num3 = 0
    for a1, a2, ans1 in pairs_list:
        if not ans1:
            continue
        for _, a3, ans2 in pairs_by_article1.get(a2, []):
            if not ans2:
                continue
            if (a1, a3) not in pair_map:
                continue
            denom3 += 1
            # check if (a1,a3) exists and get its value, otherwise False
            ans13 = pair_map.get((a1, a3), False)
            if ans13:
                num3 += 1
    prob_transitive = num3 / denom3 if denom3 > 0 else 0

    return (
        files_data,
        all_answers_count,
        overall_avg_word_length_sum,
        prob_answer_true,
        prob_cond_12_to_23,
        prob_transitive
    )


# Usage
directory_path = './data/datasets/LACD-biclassification/train-test-divide-refine'
(
    files_data,
    all_answers_count,
    overall_avg,
    p_true,
    p_23_given_12,
    p_13_given_12_23
) = calculate_statistics_and_probabilities(directory_path)

# 결과 출력
print(f"P(answer=True): {p_true:.4f}")
print(f"P(answer(a2,a3)=True | answer(a1,a2)=True): {p_23_given_12:.4f}")
print(f"P(answer(a1,a3)=True | answer(a1,a2)=True and answer(a2,a3)=True): {p_13_given_12_23:.4f}")