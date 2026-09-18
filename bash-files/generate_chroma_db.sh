# python ./src/main.py --biencoder_model_path ./data/models/LACD-bi/kbb-baseline --chroma_db_name kbb-baseline --biencoder_method baseline --retrieval_method bi-only

python ./src/main.py --biencoder_model_path ./data/models/LACD-bi/kbb-caseaug --chroma_db_name kbb-caseaug-singlecase --biencoder_method caseaug --retrieval_method bi-only


python ./src/main.py --biencoder_model_path ./data/models/LACD-bi/kbb-caseaug-multicase --chroma_db_name kbb-caseaug-multicase --biencoder_method caseaug --retrieval_method bi-only


python ./src/main.py --biencoder_model_path ./data/models/LACD-bi/kbb-baseline-finetune --chroma_db_name kbb-baseline-finetune --biencoder_method baseline --retrieval_method bi-only


python ./src/main.py --biencoder_model_path ./data/models/LACD-bi/kbb-caseaug-nofinetune --chroma_db_name kbb-caseaug-nofinetune --biencoder_method caseaug --retrieval_method bi-only


python ./src/main.py --biencoder_model_path ./data/models/LACD-bi/kbb-ruleaug --chroma_db_name kbb-ruleaug --biencoder_method ruleaug --retrieval_method bi-only

python ./src/main.py --biencoder_model_path sentence-transformers/all-MiniLM-L6-v2 --chroma_db_name all-MiniLM-L6-v2 --retrieval_method bi-only