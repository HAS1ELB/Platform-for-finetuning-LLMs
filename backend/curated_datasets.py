"""
Curated list of popular Hugging Face datasets organized by category.
This makes it easy for users to discover and use datasets without knowing technical details.
"""

CURATED_DATASETS = {
    "Sentiment Analysis & Reviews": [
        {
            "name": "stanfordnlp/imdb",
            "description": "IMDB movie reviews for sentiment analysis (25K reviews)",
            "size": "25,000 samples",
            "task": "Binary sentiment classification"
        },
        {
            "name": "yelp_review_full",
            "description": "Yelp reviews with 5-star ratings (650K reviews)",
            "size": "650,000 samples",
            "task": "5-class sentiment classification"
        },
        {
            "name": "amazon_polarity",
            "description": "Amazon product reviews (3.6M reviews)",
            "size": "3,600,000 samples",
            "task": "Binary sentiment classification"
        },
        {
            "name": "tweet_eval",
            "description": "Twitter sentiment and emotion analysis",
            "size": "Varies by subset",
            "task": "Multi-task (sentiment, emotion, etc.)"
        }
    ],
    "News & Text Classification": [
        {
            "name": "ag_news",
            "description": "AG's News articles in 4 categories (120K articles)",
            "size": "120,000 samples",
            "task": "4-class classification (World, Sports, Business, Sci/Tech)"
        },
        {
            "name": "fancyzhx/dbpedia_14",
            "description": "DBpedia ontology classification (560K samples)",
            "size": "560,000 samples",
            "task": "14-class classification"
        },
        {
            "name": "trec",
            "description": "Question classification dataset",
            "size": "5,500 samples",
            "task": "6-class question type classification"
        },
        {
            "name": "SetFit/20_newsgroups",
            "description": "Newsgroup posts in 20 categories",
            "size": "~18,000 samples",
            "task": "20-class text classification"
        }
    ],
    "Language Modeling & Text Generation": [
        {
            "name": "Salesforce/wikitext",
            "description": "Wikipedia articles for language modeling (wikitext-103)",
            "size": "103M tokens",
            "task": "Language modeling"
        },
        {
            "name": "bookcorpus",
            "description": "Books corpus for language modeling",
            "size": "~74M sentences",
            "task": "Language modeling"
        },
        {
            "name": "openwebtext",
            "description": "Web text corpus (open reproduction of GPT-2 training data)",
            "size": "~8M documents",
            "task": "Language modeling"
        },
        {
            "name": "ptb_text_only",
            "description": "Penn Treebank for language modeling",
            "size": "~1M tokens",
            "task": "Language modeling"
        }
    ],
    "Question Answering": [
        {
            "name": "rajpurkar/squad",
            "description": "Stanford Question Answering Dataset (100K questions)",
            "size": "100,000 samples",
            "task": "Extractive QA"
        },
        {
            "name": "rajpurkar/squad_v2",
            "description": "SQuAD 2.0 with unanswerable questions",
            "size": "150,000 samples",
            "task": "Extractive QA with no-answer"
        },
        {
            "name": "mandarjoshi/trivia_qa",
            "description": "Trivia questions with evidence documents",
            "size": "650,000 samples",
            "task": "Open-domain QA"
        },
        {
            "name": "allenai/sciq",
            "description": "Science questions with multiple choice answers",
            "size": "13,000 samples",
            "task": "Multiple choice QA"
        }
    ],
    "Instruction Following & Chat": [
        {
            "name": "databricks/databricks-dolly-15k",
            "description": "High-quality instruction-following dataset (15K samples)",
            "size": "15,000 samples",
            "task": "Instruction following"
        },
        {
            "name": "tatsu-lab/alpaca",
            "description": "Alpaca instruction dataset (52K samples)",
            "size": "52,000 samples",
            "task": "Instruction following"
        },
        {
            "name": "OpenAssistant/oasst1",
            "description": "Open Assistant conversation dataset",
            "size": "161,000 messages",
            "task": "Conversational AI"
        },
        {
            "name": "timdettmers/openassistant-guanaco",
            "description": "Curated subset of Open Assistant for fine-tuning",
            "size": "~10,000 samples",
            "task": "Instruction following & chat"
        },
        {
            "name": "HuggingFaceH4/ultrachat_200k",
            "description": "High-quality synthetic chat conversations",
            "size": "200,000 samples",
            "task": "Conversational AI"
        }
    ],
    "Summarization": [
        {
            "name": "abisee/cnn_dailymail",
            "description": "CNN/Daily Mail news articles with summaries (300K pairs)",
            "size": "300,000 samples",
            "task": "Abstractive summarization"
        },
        {
            "name": "EdinburghNLP/xsum",
            "description": "Extreme summarization of BBC articles",
            "size": "227,000 samples",
            "task": "Extreme abstractive summarization"
        },
        {
            "name": "Samsung/samsum",
            "description": "Dialogue summarization dataset",
            "size": "16,000 samples",
            "task": "Dialogue summarization"
        },
        {
            "name": "multi_news",
            "description": "Multi-document summarization",
            "size": "56,000 samples",
            "task": "Multi-document summarization"
        }
    ],
    "Translation": [
        {
            "name": "wmt14",
            "description": "WMT 2014 translation datasets (multiple language pairs)",
            "size": "Varies by language pair",
            "task": "Machine translation"
        },
        {
            "name": "Helsinki-NLP/opus-100",
            "description": "100 language pairs from OPUS corpus",
            "size": "Varies by language pair",
            "task": "Multilingual translation"
        }
    ],
    "Code": [
        {
            "name": "code_search_net",
            "description": "Code documentation dataset (2M code-comment pairs)",
            "size": "2,000,000 samples",
            "task": "Code understanding & generation"
        },
        {
            "name": "bigcode/the-stack-dedup",
            "description": "Deduplicated version of The Stack code dataset",
            "size": "~3TB of code",
            "task": "Code generation"
        },
        {
            "name": "mbpp",
            "description": "Mostly Basic Python Problems for code generation",
            "size": "974 samples",
            "task": "Python code generation"
        }
    ],
    "Common Small Datasets": [
        {
            "name": "sst2",
            "description": "Stanford Sentiment Treebank (binary classification)",
            "size": "67,000 samples",
            "task": "Sentiment analysis"
        },
        {
            "name": "rotten_tomatoes",
            "description": "Movie review sentiment (short texts)",
            "size": "10,600 samples",
            "task": "Binary sentiment classification"
        },
        {
            "name": "SetFit/sst5",
            "description": "Stanford Sentiment Treebank (5-class)",
            "size": "11,800 samples",
            "task": "Fine-grained sentiment"
        },
        {
            "name": "glue",
            "description": "General Language Understanding Evaluation benchmark",
            "size": "Varies by task",
            "task": "Multiple NLU tasks"
        },
        {
            "name": "paws",
            "description": "Paraphrase detection dataset",
            "size": "108,000 samples",
            "task": "Paraphrase identification"
        }
    ]
}

def get_all_datasets_flat():
    """Get a flat list of all curated datasets."""
    all_datasets = []
    for category, datasets in CURATED_DATASETS.items():
        for dataset in datasets:
            dataset_copy = dataset.copy()
            dataset_copy["category"] = category
            all_datasets.append(dataset_copy)
    return all_datasets

def get_dataset_by_name(name: str):
    """Find a dataset by name in the curated list."""
    for category, datasets in CURATED_DATASETS.items():
        for dataset in datasets:
            if dataset["name"] == name:
                dataset_copy = dataset.copy()
                dataset_copy["category"] = category
                return dataset_copy
    return None

def search_datasets(query: str):
    """Search datasets by name or description."""
    query = query.lower()
    results = []
    for category, datasets in CURATED_DATASETS.items():
        for dataset in datasets:
            if (query in dataset["name"].lower() or 
                query in dataset["description"].lower() or
                query in category.lower()):
                dataset_copy = dataset.copy()
                dataset_copy["category"] = category
                results.append(dataset_copy)
    return results
