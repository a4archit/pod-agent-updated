# Pod-agent


An agent that will able to generate quiz on the basis of uploaded PDF.






### Codebase Architecture (local)
```text
├── file.pdf
├── local_dev.md
├── main.py
├── podagent
│   ├── agent.py
│   ├── configs.py
│   ├── faiss_manager.py
│   ├── pdf_processor.py
│   ├── qdrant_manager.py
│   ├── rag.py
│   ├── scripts.ipynb
│   └── temp_faiss
│       └── vector_faiss
│           ├── index.faiss
│           └── index.pkl
├── README.md
├── requirements.txt
└── temp_faiss
    └── vector_faiss
        ├── index.faiss
        └── index.pkl
```






