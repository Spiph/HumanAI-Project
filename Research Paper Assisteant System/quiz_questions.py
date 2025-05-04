# quiz_questions.py

PAPER_QUIZ = {
    "Are You the Main Character? Visibility Labor and Attributional Practices on TikTok": [
        {
            "question": "Which three explanatory frameworks does Losh integrate in this paper?",
            "choices": [
                ("Hypertext theory, database cinema, visibility labor", True),
                ("Interface design, recommendation algorithms, citation analysis", False),
                ("Network analysis, algorithmic governance, privacy studies", False),
                ("Participatory culture, platform economics, digital literacy", False),
            ],
        },
        {
            "question": "Which TikTok affordance automates hyperlinking, creating rich attributional practices of citation?",
            "choices": [
                ("Manual tagging of usernames in captions", False),
                ("“Duet” and “Stitch” functions when selecting audio clips, effects, and hashtags", True),
                ("Comment threads linking related videos", False),
                ("Live‑streaming with real‑time metadata overlays", False),
            ],
        },
        {
            "question": "In Losh’s paradigm of “visibility labor,” what work are creators performing?",
            "choices": [
                ("Designing TikTok’s recommendation algorithm", False),
                ("Adding metadata (hashtags, audio, effects) to ensure their content remains discoverable", True),
                ("Investing in paid promotions to boost views", False),
                ("Moderating community comments on their videos", False),
            ],
        },
        {
            "question": "What term does Losh use to describe creators’ efforts to withdraw or obfuscate their own content from TikTok’s hyperlinked matrix?",
            "choices": [
                ("Shadow banning", False),
                ("Invisibility labor", True),
                ("Content moderation", False),
                ("Ghost work", False),
            ],
        },
        {
            "question": "Who originally defined hypertext as “forms of writing which branch or perform on request,” a definition Losh revisits in this paper?",
            "choices": [
                ("George Landow", False),
                ("Ted Nelson", True),
                ("Lev Manovich", False),
                ("Jay David Bolter", False),
            ],
        },
    ],

    "Effects of the Spiral of Silence on Minority Groups in Recommender Systems": [
        {
            "question": "What primary theoretical phenomenon does the paper examine as contributing to minority underrepresentation in recommender systems?",
            "choices": [
                ("Filter bubbles", False),
                ("Spiral of silence", True),
                ("Cold‑start problem", False),
                ("Confirmation bias", False),
            ],
        },
        {
            "question": "According to the spiral of silence theory described in Section 2.1, minority users are likely to…",
            "choices": [
                ("Increase their rating frequency", False),
                ("Withhold their true opinions due to fear of social isolation", True),
                ("Use multiple accounts for anonymity", False),
                ("Provide more detailed feedback", False),
            ],
        },
        {
            "question": "The empirical findings by Liu et al. [18] cited in Section 3.2 indicate that missing ratings in recommender systems follow a…",
            "choices": [
                ("Random pattern", False),
                ("Uniform distribution", False),
                ("Non‑random pattern linked to minority silence", True),
                ("Normal distribution", False),
            ],
        },
        {
            "question": "Which fairness‑centric model do the authors mention as integrating fairness for both users and items in recommender systems?",
            "choices": [
                ("FAiR", True),
                ("CF‑Plus", False),
                ("EquityRec", False),
                ("BiasNeg", False),
            ],
        },
        {
            "question": "As a future research direction (Section 5), the paper suggests designing user interfaces that…",
            "choices": [
                ("Hide minority opinions to prevent bias", False),
                ("Represent minority views alongside popular opinions to encourage expression", True),
                ("Automatically anonymize all user inputs", False),
                ("Replace recommendations with random selections", False),
            ],
        },
    ],

    "Profiling Fake News Spreaders on Social Media through Psychological and Motivational Factors": [
        {
            "question": "Which tool did the authors use to extract psychologically‐relevant word categories (e.g., anxiety, tentativeness) from users’ tweets?",
            "choices": [
                ("Word2Vec", False),
                ("LIWC", True),
                ("NLTK", False),
                ("GloVe", False),
            ],
        },
        {
            "question": "The study draws on two datasets from the FakeNewsNet repository. Which are they?",
            "choices": [
                ("BuzzFeed and Snopes", False),
                ("PolitiFact and GossipCop", True),
                ("Reuters and AP News", False),
                ("FactCheck.org and Media Bias/Fact Check", False),
            ],
        },
        {
            "question": "To quantify the “lack of control” motivational factor, the authors measured the proportion of words in which LIWC category?",
            "choices": [
                ("Certainty", False),
                ("Anxiety", False),
                ("Future Focus", True),
                ("Tentativeness", False),
            ],
        },
        {
            "question": "Which pre‑trained embedding model did the authors augment with their motivational‑feature vector for the fake‑spreader classification task?",
            "choices": [
                ("GPT‑2", False),
                ("ELMo", False),
                ("BERT", True),
                ("FastText", False),
            ],
        },
        {
            "question": "What dimensionality‑reduction technique did they use to visualize the difference between BERT and BERT+Features embeddings?",
            "choices": [
                ("PCA", False),
                ("t‑SNE", True),
                ("UMAP", False),
                ("LDA", False),
            ],
        },
    ],

    "Emotional Hermeneutics. Exploring the Limits of Artificial Intelligence from a Diltheyan Perspective": [
        {
            "question": "According to Picca’s summary of Dilthey’s hermeneutics, “understanding” in the human sciences is characterized by:",
            "choices": [
                ("Reduction of phenomena into causal laws", False),
                ("Introspection and interpretative (hermeneutic) engagement with context", True),
                ("Quantitative measurement of emotional responses", False),
                ("Algorithmic pattern recognition of text", False),
            ],
        },
        {
            "question": "What key limitation of large language models (LLMs) does Picca highlight when applying Dilthey’s “understanding” to AI?",
            "choices": [
                ("Insufficient training data size", False),
                ("Lack of personal experiences and self‑awareness for genuine emotional comprehension", True),
                ("Inability to perform fast data‑driven explanations", False),
                ("Overreliance on symbolic logic", False),
            ],
        },
        {
            "question": "Picca suggests enhancing AI’s emotional interpretative capacity by integrating insights from which fields?",
            "choices": [
                ("Quantum physics and mathematics", False),
                ("Humanities and social sciences", True),
                ("Pure computational theory", False),
                ("Financial modeling and economics", False),
            ],
        },
        {
            "question": "Which of the following does Picca identify as an “insurmountable limit” for AI attempting true emotional hermeneutics?",
            "choices": [
                ("Algorithmic speed constraints", False),
                ("Absence of self‑reflection and subjective, lived experience", True),
                ("Limited hardware memory", False),
                ("Lack of internet connectivity", False),
            ],
        },
        {
            "question": "In the context of emotional AI’s future, Picca notes that “emergent properties” in more advanced architectures might:",
            "choices": [
                ("Eliminate the need for human‑machine interaction", False),
                ("Allow AI to develop novel, unanticipated capabilities that could approximate deeper emotional understanding", True),
                ("Reduce all emotional data to single numeric scores", False),
                ("Ensure perfect replication of human feelings", False),
            ],
        },
    ],

    "Exercises in Unimaginativeness: Case Study of GPT-Based Translation and Travesty of Alfred Jarry's \"Ubu King\"": [
        {
            "question": "What theoretical notion does “uncreative writing” emphasize in Argasiński & Marecki’s case study?",
            "choices": [
                ("Generating entirely novel text from scratch", False),
                ("Repurposing and transforming existing texts as data", True),
                ("Human‑only authorship without machine assistance", False),
                ("Eliminating all stylistic markers from the source", False),
            ],
        },
        {
            "question": "Which LLM did the authors use to translate Ubu Roi into Polish and then re‑interpret it?",
            "choices": [
                ("GPT‑2", False),
                ("GPT‑3.5", False),
                ("GPT‑4o", True),
                ("BERT", False),
            ],
        },
        {
            "question": "How did the script ensure that character names remained consistent across the translation?",
            "choices": [
                ("By hard‑coding them post‑translation in a spreadsheet", False),
                ("By including the full list of characters in every prompt to GPT‑4o", True),
                ("By manually editing each scene in Google Docs", False),
                ("By using an external name‑normalization API", False),
            ],
        },
        {
            "question": "Which GAN model was employed to generate the book’s illustrations?",
            "choices": [
                ("StyleGAN2", False),
                ("CycleGAN", False),
                ("BigGAN", False),
                ("DALL·E 3", True),
            ],
        },
        {
            "question": "In what format was the final project published to bridge digital experimentation and traditional media?",
            "choices": [
                ("As a downloadable PDF on the authors’ website", False),
                ("As an interactive web app", False),
                ("As a traditionally‑printed book with an ISBN and bookstore distribution", True),
                ("As a limited NFT art release", False),
            ],
        },
    ],
}
