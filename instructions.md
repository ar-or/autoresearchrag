
Take the code
- postulate an improvement hypothesis to the rag pipeline. Something you want to try that can improve the overall performance.
- fork develop into a new git branch: feature_<hypothesis>
- clone es index `mtrag` to feature_<hypothesis>
- set environment variables N=30 ; ES_INDEX=feature_<hypothesis>
- make changes to  agent.py or scripts/ingest.py according to the improvement hypothesis
- Don't change the embedding model or the fact that embedding is resolved with sqlite cache.
- if the ingestion changed in any way, delete from ES_INDEX index all documents related to finqa (where colelction_name=mycollection1) and reingest them with `evaluators/finqa/ingest_finqa.py`
- run finqa evaluation with N=30 (but not mtrag and not hot hotpot and not any other evaluator)
- record the score in results.csv 
- merge results.csv into develop branch
- if the finqa overall result better than best_result.txt -> override best_result.txt and merge current branch into develop so next experiment will start with this current changes as baseline

