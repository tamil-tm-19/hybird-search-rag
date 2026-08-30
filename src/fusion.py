from typing import List , Dict, Any , Tuple

def rrf(vectors_result: List[List[Any]], 
        top_k: int = 5):

    rrf_result = {}
    document = {}
    k = 60

    try:
        
        for vectors in vectors_result:
            for rank , vector in enumerate(vectors, start = 1):
                score = 1 / (k + rank)
                rrf_result[vector['id']] = rrf_result.get(vector['id'] , 0) + score

                if vector['id'] not in document:
                    document[vector['id']] = vector['text']

                sort_result = sorted(rrf_result.items() , key = lambda x : x[1] , reverse = True)


            finall_rrf_result : List[str] = []

            for result in sort_result:
                finall_rrf_result.append(document[result[0]])
            return finall_rrf_result[:top_k]
        
    except Exception as e:
        print(f"ERROR : {e}")
        raise e