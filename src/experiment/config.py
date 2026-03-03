"""
List of possible algorithm types:
    1) naive - scipy realization with uploading data;
    2) iterative - spark k-splits single calculation realization for single split without uploading data; 
    3) parallel - spark k-splits single calculation realization without uploading data;
    4) alternative - spark alternative realization without uploading data. 

List of possible tests:
    1) ttest - two samples t-test;
    2) chisquare - two samples chi-square test;
    3) kstest - two samples ks-test.
"""

test_list = [
    # {
    #     "ttest_01" : {
    #         "test_type" : ["kstest"],
    #         "algo_type" : [
    #                             "naive",
    #                             "iterative",
    #                             # "parallel"
    #                         ]
    #     }
    # },
    # {
    #     "chisquare_01" : {
    #         "test_type" : ["chisquare"],
    #         "algo_type" : [
    #                             "naive",
    #                             "iterative",
    #                             "parallel",
    #                         ]
    #     }
    # },
    # {
    #     "ks_01" : {
    #         "test_type" : ["ttest"],
    #         "algo_type" : [
    #                             # "naive",
    #                             # "iterative",
    #                             "alternative",
    #                             "parallel",
                                
    #                         ]
    #     }
    # },
    {
        "all_tests_01" : {
            "test_type" : [
                                "chisquare", 
                                "ttest", 
                                "kstest"
                            ],
            "algo_type" : [
                                "naive",
                                "iterative",
                                # "parallel",
                            ]
        }
    },
    # {
    #     "all_tests_pandas_01" : {
    #         "test_type" : [
    #                             "chisquare", 
    #                             "ttest", 
    #                             # "kstest"
    #                         ],
    #         "algo_type" : [
    #                             "iterative",
    #                             "parallel",
    #                         ]
    #     }
    # },
    # {
    #     "ks_test_01" : {
    #         "test_type" : [
    #                             "kstest"
    #                         ],
    #         "algo_type" : [
    #                             "naive",
    #                             "iterative",
    #                             "alternative"
    #                         ]
    #     }
    # },
]

experiment_data = {
    "constant var" : {
        "random_state" : 21,
        "groups_num" : 4,
        "target_category_columns" : ["industry", "gender"],
        "target_numeric_columns" : ["pre_spends", "post_spends", "age"]
    } ,
    "variative var" : {
        "k_splits" : [2],
        "fractions" : [1],
        "target_frac" : [1], # Доля от всех фич, которые могут быть использованы
    }
}