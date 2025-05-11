# Rayyaan Haamid - AI & Deep Learning Portfolio (ITAI 2376)

**Last Updated:** May 11, 2025

## Welcome! 👋

This repository showcases my comprehensive learning journey and project work from the **ITAI 2376: Deep Learning and AI Agent Development** course. Here, you'll find a collection of my assignments, labs, and projects, demonstrating a progressive understanding and application of core AI, Machine Learning, and Deep Learning concepts—from foundational principles to the development of sophisticated AI agent systems.

My goal throughout this course was not just to learn, but to **build** and **apply**. This portfolio is a testament to that hands-on approach, culminating in a capstone project that integrates many of the skills and technologies explored.

**🔗 View My Portfolio Presentation:** https://www.canva.com/design/DAGnKD0F4bs/gXgPYMuU4D2BTcsNHGbm9A/view?utm_content=DAGnKD0F4bs&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hca14ecd72a

---

🌟 ## Portfolio Highlights

This repository is structured to reflect the progression of the course, with key highlights including:

### ☁️ Foundational AI with Cloud Services (Azure)
- Hands-on labs using Azure AI services for:
    - Content Safety & Moderation
    - Computer Vision (Image Analysis, Object Detection with Azure Vision Studio)
    - Natural Language Processing (Sentiment Analysis, Key Phrase Extraction with Azure Language Studio)
    - Document Intelligence & OCR
    - Generative AI Prompt Engineering with Azure AI Foundry
- **Key Skills:** Azure resource management, utilizing pre-trained models, API integration, understanding practical AI applications.
- *(Explore folders: `L01_Azure_Content_Safety`, `L02_Azure_Vision`, `L03_Azure_Language`, `L04_Azure_Document_Intelligence`, `L05_Azure_Generative_AI`)*

### 🧠 Deep Learning Fundamentals
- Exploration of deep learning concepts via no-code (VGG16 with TensorFlow/Keras) and coded approaches.
- Implementation of Convolutional Neural Networks (CNNs) for image classification (MNIST - **achieving 99.12% accuracy**).
- Creative exploration of various neural network architectures (CNNs, RNNs, LSTMs, GANs, Transformers) through the "Neural Network Zoo" project (e.g., CNNs as Falcons, Transformers as Owls).
- Developing simplified explanations of complex topics like Gradient Descent for a non-technical audience.
- *(Explore folders: `L02_NoCode_DL`, `L04_CNN_MNIST`, `A03_Neural_Network_Zoo`, `A04_DL_For_11_Year_Old`)*

### 💬 Advanced NLP and Transformer Architectures
- In-depth labs on text processing pipelines (stemming, lemmatization, POS tagging, NER), Bag-of-Words, TF-IDF, and word embeddings (GloVe) using AWS MLU.
- Practical experience with Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks for sequential data.
- Fine-tuning **BERT** (a Transformer model) for sentiment analysis (**~92% accuracy achieved**), demonstrating understanding of tokenization, attention mechanisms, and transfer learning.
- Comparative analysis of NLP tools (Hugging Face Transformers vs. OpenAI GPT-4 API).
- Critical analysis of NLP challenges through the lens of the film "Arrival."
- *(Explore folders: `L03_AWS_MLU_PyTorch`, `L05_AWS_MLU_NLP`, `L06_AWS_MLU_BERT`, `A02_Tool_Comparison`, `A05_Arrival_NLP_Analysis`)*

### 🎨 Generative AI - Diffusion Models (Midterm Project)
- End-to-end implementation and training of a **U-Net based diffusion model** to generate Fashion-MNIST images from scratch.
- Incorporated time and class conditioning, and visualized the step-by-step denoising process.
- Implemented optional **CLIP evaluation** to quantitatively assess generated image quality against text prompts (achieving scores ~0.26-0.28).
- *(Explore folder: `Midterm_Diffusion_Models` - includes notebook PDF and analysis report)*

### 🏆 Capstone Project - "Health Advisor" AI Agent
- **Solo development** of a full-stack, multi-tool AI agent providing personalized health and well-being advice.
- **Architecture:** Azure GPT-4o-mini integrated with LangChain, employing **ReAct** and **Plan-and-Execute** agent patterns.
- **Tech Stack:** Python (Flask) backend, Next.js frontend, Supabase (Postgres) for persistent memory.
- **Features:** Utilized **14+ tools** including health calculators, information retrieval from APIs (WGER, MealDB, OpenFoodFacts), web search (SerpAPI/DuckDuckGo), offline databases, and a FAISS PDF retriever.
- **RL Element:** Implemented an "RL-Lite" feedback loop where user interactions implicitly reward successful tool chains, with a simulated nightly cron job adjusting tool selection probabilities (e.g., barcode lookup tool prior increased from 0.06 to 0.14 after 50 'saves', reducing wrong-tool calls by 32%).
- **Safety:** Incorporated input validation, profanity filtering, a `guardian_tool` for sensitive queries, sandboxed API calls with domain whitelisting, and graceful error handling.
- *(Explore folder: `Capstone_Health_Advisor` - includes project report, link to the code repository (if separate), and demo video link)*

---

*(Note: File names reflect my solo work for Midterm & Capstone where applicable, and group work for others. Ensure your file names in the repo match this structure or update accordingly.)*

---

🛠️ ## Technologies & Tools Leveraged

Throughout these projects, I gained experience with a diverse set of tools and technologies, including:

- **Cloud Platforms:** Microsoft Azure AI Studio (Content Safety, Vision, Language, Document Intelligence, AI Foundry, GPT-4o-mini), AWS Machine Learning University Labs.
- **Programming Languages:** Python
- **Core Libraries:** PyTorch, TensorFlow, Keras, LangChain, Scikit-learn, Pandas, NumPy, Matplotlib, NLTK, spaCy, Transformers (Hugging Face).
- **APIs & Services:** OpenAI API (conceptual), SerpAPI, DuckDuckGo Search, WGER API, MealDB API, OpenFoodFacts API, Supabase (Postgres).
- **Development Environments:** Jupyter Notebooks, Google Colab, Flask, Next.js.
- **Techniques & Architectures:** CNNs, RNNs, LSTMs, Transformers (BERT), Diffusion Models (U-Net), ReAct Agent Pattern, Plan-and-Execute Pattern, TF-IDF, GloVe, WordPiece Tokenization, FAISS, CLIP.

---

🚀 ## Key Learnings & Reflections

This course has been instrumental in shaping my understanding of AI from multiple perspectives:

1.  **From Theory to Practice:** The hands-on nature of the labs and projects was vital in solidifying theoretical knowledge. Building models and agents from scratch or fine-tuning existing ones provided invaluable insights that lectures alone cannot offer.
2.  **The Power of Iteration:** AI development is rarely a linear process. I learned the importance of iterative development, debugging, and refining solutions based on empirical results and encountered challenges (e.g., pivoting from Semantic Kernel JS in the Capstone).
3.  **Tooling Landscape:** The AI field is rich with powerful tools and frameworks. Gaining familiarity with leaders like Azure AI, Hugging Face Transformers, and LangChain has equipped me to tackle diverse problems.
4.  **Agentic AI Design:** The Capstone project, in particular, provided a deep dive into designing AI agents with distinct components for input processing, memory, reasoning (ReAct, Planning), and tool use, along with safety considerations.
5.  **Ethical AI & Safety:** Assignments like the "Arrival" analysis and the safety implementations in the Capstone project underscored the critical importance of considering ethical implications, biases, and safety boundaries in AI systems.

---

💡 ## Future Directions & Interests

My experience in ITAI 2376 has fueled my interest in several areas of AI, including:

- Advanced Multi-Agent Systems and Collaboration.
- Deep Reinforcement Learning for more adaptive and robust agent behavior.
- Multimodal AI, capable of understanding and generating content across different data types.
- Contributing to the open-source AI community.
- Applying AI to solve real-world problems, particularly in areas like personalized education and health/well-being.

---

🙏 ## Acknowledgements

Thank you to the instructors and TAs of ITAI 2376 for their guidance and for crafting a curriculum that balanced theory with extensive practical application.

---

📬 ## Connect With Me

- **GitHub:**
- **Email:** rayyaan.haamid@gmail.com

Thank you for visiting my portfolio! I hope it provides a clear view of my skills, dedication, and growth in the fascinating field of Artificial Intelligence.
