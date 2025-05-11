# Portfolio

Rayyaan Haamid - AI & Deep Learning Portfolio (ITAI 2376)
Last Updated: May 11, 2025

👋 Welcome!
This repository showcases my comprehensive learning journey and project work from the ITAI 2376: Deep Learning and AI Agent Development course. Here, you'll find a collection of my assignments, labs, and projects, demonstrating a progressive understanding and application of core AI, Machine Learning, and Deep Learning concepts—from foundational principles to the development of sophisticated AI agent systems.

My goal throughout this course was not just to learn, but to build and apply. This portfolio is a testament to that hands-on approach, culminating in a capstone project that integrates many of the skills and technologies explored.

🔗 View My Portfolio Presentation: [Link to Your Pf_Rayyaan_Haamid_ITAI2376.pdf Presentation Here]

🌟 Portfolio Highlights
This repository is structured to reflect the progression of the course, with key highlights including:

Foundational AI with Cloud Services (Azure):

Hands-on labs using Azure AI services for content safety, computer vision (image analysis, object detection), natural language processing (sentiment analysis, key phrase extraction), document intelligence, and generative AI prompt engineering with Azure AI Foundry.
Key Skills: Azure resource management, utilizing pre-trained models, API integration, understanding practical AI applications.
(See folders: L01_Azure_Content_Safety, L02_Azure_Vision, L03_Azure_Language, L04_Azure_Document_Intelligence, L05_Azure_Generative_AI)
Deep Learning Fundamentals:

Exploration of deep learning concepts via no-code (VGG16 with TensorFlow/Keras) and coded approaches.
Implementation of Convolutional Neural Networks (CNNs) for image classification (MNIST - achieving 99.12% accuracy).
Creative exploration of various neural network architectures (CNNs, RNNs, LSTMs, GANs, Transformers) through the "Neural Network Zoo" project.
Developing simplified explanations of complex topics like Gradient Descent for a non-technical audience.
(See folders: L02_NoCode_DL, L04_CNN_MNIST, A03_Neural_Network_Zoo, A04_DL_For_11_Year_Old)
Advanced NLP and Transformer Architectures:

In-depth labs on text processing pipelines (stemming, lemmatization, POS tagging, NER), Bag-of-Words, TF-IDF, and word embeddings (GloVe).
Practical experience with Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks for sequential data.
Fine-tuning BERT (a Transformer model) for sentiment analysis, demonstrating understanding of tokenization, attention mechanisms, and transfer learning.
Comparative analysis of NLP tools (Hugging Face Transformers vs. OpenAI GPT-4 API).
Critical analysis of NLP challenges through the lens of the film "Arrival."
(See folders: L03_AWS_MLU_PyTorch, L05_AWS_MLU_NLP, L06_AWS_MLU_BERT, A02_Tool_Comparison, A05_Arrival_NLP_Analysis)
Generative AI - Diffusion Models (Midterm Project):

End-to-end implementation and training of a U-Net based diffusion model to generate Fashion-MNIST images from scratch.
Incorporated time and class conditioning, and visualized the step-by-step denoising process.
Implemented optional CLIP evaluation to quantitatively assess generated image quality against text prompts (achieving scores ~0.26-0.28).
(See folder: Midterm_Diffusion_Models - includes notebook and analysis report)
Capstone Project - "Health Advisor" AI Agent:

Solo development of a full-stack, multi-tool AI agent providing personalized health and well-being advice.
Architecture: Azure GPT-4o-mini integrated with LangChain, employing ReAct and Plan-and-Execute agent patterns.
Tech Stack: Python (Flask) backend, Next.js frontend, Supabase (Postgres) for persistent memory.
Features: Utilized 14+ tools including health calculators, information retrieval from APIs (WGER, MealDB, OpenFoodFacts), web search (SerpAPI/DuckDuckGo), offline databases, and a PDF retriever (FAISS).
RL Element: Implemented an "RL-Lite" feedback loop where user interactions (saving/deleting advice) implicitly reward successful tool chains, with a simulated nightly cron job adjusting tool selection probabilities.
Safety: Incorporated input validation, profanity filtering, a guardian_tool for sensitive queries, sandboxed API calls with domain whitelisting, and graceful error handling.
(See folder: Capstone_Health_Advisor - includes project report, link to code repository, and demo video link)
📂 Repository Structure
This repository is organized by modules and assignments for easy navigation:

.
├── L01_Azure_Content_Safety/
│   └── L01_Rayyaan_Haamid_ITAI2372.pdf
├── L02_Azure_Vision/
│   └── L02_Rayyaan_Haamid_ITAI2372.pdf
├── L03_Azure_Language/
│   └── L03_Rayyaan_Haamid_ITAI2372.pdf
├── L04_Azure_Document_Intelligence/
│   └── L04_Rayyaan_Haamid_ITAI2376.pdf
├── L05_Azure_Generative_AI/
│   └── L05_Rayyaan_Haamid_ITAI2376.pdf
├── L02_NoCode_DL/
│   └── L02_Rayyaan_Haamid_ITAI2376.pdf
├── L03_AWS_MLU_PyTorch/
│   └── L03AWSMLU_Rayyaan_Haamid_ITAI_2376.pdf
├── L04_CNN_MNIST/
│   └── Lab04_CNN_Rayyaan_Haamid_ITAI2676.pdf
├── L05_AWS_MLU_NLP/
│   └── L05_Journal_Rayyaan_Haamid_ITAI22376.pdf
├── L06_AWS_MLU_BERT/
│   └── L06_Journal_Rayyaan_Haamid_ITAI22376.pdf
├── A02_Tool_Comparison/
│   ├── A02a_Neural_Titians_SubmitterName_ITA2376.pptx
│   ├── A02b_NeuroTitans_MustafaYucedag_ITA2376.pdf
│   └── A02_huggingFace_vs_GPT4.ipynb
├── A03_Neural_Network_Zoo/
│   └── A03_Neural_Titans_Rayyaan_Haamid_ITAI2376.pdf
├── A04_DL_For_11_Year_Old/
│   ├── A04_Neural_Titans_Alhassane_Samassekou_ITAI2376.docx
│   └── A04_Neural_Titans_Alhassane_Samassekou_ITAI2376.pptx
├── A05_Arrival_NLP_Analysis/
│   ├── A05_Neural_Titans_Rayyaan_Haamid_ITAI2376.pdf
│   └── A05_NeuralTitans_Mustafa-Yucedag_ITAI2376.docx
├── Midterm_Diffusion_Models/
│   ├── Midterm_Diffusion_Model_Rayyaan_Haamid.pdf      # Analysis Report
│   └── Diffusion_FashionMNIST_Kaggle_Colab.pdf         # Notebook Output
├── Capstone_Health_Advisor/
│   ├── FN_Report_Rayyaan_Haamid_ITAI2376.pdf           # Project Report
│   ├── FN_Code_Rayyaan_Haamid_Solo_ITAI2376.zip        # Or link to separate repo
│   └── FN_Demo_Rayyaan_Haamid_Solo_ITAI2376.mp4        # Link to video
├── Pf_Rayyaan_Haamid_ITAI2376.pdf                      # This Portfolio Presentation
└── README.md                                           # This file
(Note: Some filenames above are illustrative based on common group submissions; I've adjusted Capstone and Midterm to reflect my solo work as described in the reports.)

🛠️ Technologies & Tools Leveraged
Throughout these projects, I gained experience with a diverse set of tools and technologies, including:

Cloud Platforms: Microsoft Azure AI Studio (Content Safety, Vision, Language, Document Intelligence, AI Foundry, GPT-4o-mini), AWS Machine Learning University Labs.
Programming Languages: Python
Core Libraries: PyTorch, TensorFlow, Keras, LangChain, Scikit-learn, Pandas, NumPy, Matplotlib, NLTK, spaCy, Transformers (Hugging Face).
APIs & Services: OpenAI API (conceptual understanding), SerpAPI, DuckDuckGo Search, WGER API, MealDB API, OpenFoodFacts API, Supabase (Postgres).
Development Environments: Jupyter Notebooks, Google Colab, Flask, Next.js.
Techniques & Architectures: CNNs, RNNs, LSTMs, Transformers (BERT), Diffusion Models (U-Net), ReAct Agent Pattern, Plan-and-Execute Pattern, TF-IDF, GloVe, WordPiece Tokenization, FAISS.
🚀 Key Learnings & Reflections
This course has been instrumental in shaping my understanding of AI from multiple perspectives:

From Theory to Practice: The hands-on nature of the labs and projects was vital in solidifying theoretical knowledge. Building models and agents from scratch or fine-tuning existing ones provided invaluable insights that lectures alone cannot offer.
The Power of Iteration: AI development is rarely a linear process. I learned the importance of iterative development, debugging, and refining solutions based on empirical results and encountered challenges (e.g., pivoting from Semantic Kernel JS in the Capstone).
Tooling Landscape: The AI field is rich with powerful tools and frameworks. Gaining familiarity with leaders like Azure AI, Hugging Face Transformers, and LangChain has equipped me to tackle diverse problems.
Agentic AI Design: The Capstone project, in particular, provided a deep dive into designing AI agents with distinct components for input processing, memory, reasoning (ReAct, Planning), and tool use, along with safety considerations.
Ethical AI & Safety: Assignments like the "Arrival" analysis and the safety implementations in the Capstone project underscored the critical importance of considering ethical implications, biases, and safety boundaries in AI systems.
💡 Future Directions & Interests
My experience in ITAI 2376 has fueled my interest in several areas of AI, including:

Advanced Multi-Agent Systems and Collaboration.
Deep Reinforcement Learning for more adaptive and robust agent behavior.
Multimodal AI, capable of understanding and generating content across different data types (text, image, audio).
Contributing to the open-source AI community.
Applying AI to solve real-world problems, particularly in areas like personalized education and health/well-being.
🙏 Acknowledgements
Thank you to the instructors and TAs of ITAI 2376 for their guidance and for crafting a curriculum that balanced theory with extensive practical application.

📬 Connect With Me
GitHub: [Your GitHub Profile Link]
LinkedIn: [Your LinkedIn Profile URL (Optional)]
Email: [Your Email Address]
Thank you for visiting my portfolio! I hope it provides a clear view of my skills, dedication, and growth in the fascinating field of Artificial Intelligence.
