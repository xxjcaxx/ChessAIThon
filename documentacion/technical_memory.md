## Introduction

Welcome to the technical documentation for the ChessAIThon project, an initiative designed to bridge the gap between vocational education and the evolving demands of the digital workforce. This technical memory specifically details the software development phase, outlining the methodologies, tools, and processes employed to bring the project's innovative vision to life.

The ChessAIThon project, deeply rooted in the belief that strategic gaming like chess can cultivate essential skills such as critical thinking, pattern recognition, and decision-making, aims to integrate the practical application of Artificial Intelligence (AI) into Vocational Education and Training (VET) curricula. Our consortium recognized the critical need for VET institutions to adapt and equip students with robust coding, data management, and AI competencies, areas often overlooked in traditional IT curricula.

This project directly addresses the VET sector's priority of "Adapting vocational education and training to labour market needs" by providing a balanced mix of vocational skills through strategic gaming and understanding AI implementation. Furthermore, it contributes to "Innovation in vocational education and training" by engaging students in a dynamic learning environment that applies AI to chess through an interactive interface. The project also aligns with the "Addressing digital transformation through development of digital readiness, resilience and capacity" horizontal priority, stimulating the development of digital pedagogical skills by purposefully incorporating digital technologies, AI, and chess.

The core of ChessAIThon lies in its unconventional approach: integrating chess logic into the curriculum to develop foundational skills necessary for using AI. The logical and algorithmic thinking inherent in chess seamlessly aligns with the cognitive processes integral to coding and data analysis, making it an ideal complement to traditional programming education.

This document will detail the development of key components, including:

    A comprehensive online database: This dynamic repository will feature real-life chess scenarios and allow users to propose solutions, ultimately aggregating student and player contributions to train an AI. The system will leverage the Chess.js library for displaying legal moves and enable data export for archiving and version control.

The Chess Artificial Intelligence Hackathon platform: This innovative competition will challenge students to use automated tools or create their own to train an AI with their documented chess moves, fostering a holistic understanding of chess strategy, computational thinking, data structures, version control, AI models, and cloud computing.

By focusing on these practical applications, ChessAIThon aims to provide VET students with a more comprehensive learning experience that goes beyond basic programming, incorporating essential aspects such as effective data management, version control, and a deeper exploration of AI concepts not commonly encountered at this educational level. This technical memory serves as a testament to the project's commitment to preparing students for the technical demands of contemporary workplaces and fostering a new generation of skilled professionals in AI and frontend development.


## Objectives 


**Curriculum Integration & Teacher Support**: Provide tools to integrate chess-based learning for developing critical thinking, problem-solving, and creativity. This includes functionalities for teaching coding principles through chess scenarios, fundamental AI and Machine Learning concepts, data structures, and the utilization of public datasets like Kaggle for AI training. The software will also facilitate teaching the use of Large Language Models (LLMs) for chess and coding problems, and version control for chess scenarios.

**Interactive Learning Platform**: Develop an online platform with a visually appealing chessboard interface. This platform will enable users to propose and engage with real-life chess scenarios. It will utilize the Chess.js library for displaying all legal moves, validating user moves, and storing valid moves in a database.

**Data Management & Analysis**: Implement robust data handling capabilities, including the representation of chess with various computer file formats like Portable Game Notation (PGN), JSON, CSV, Forsyth-Edwards Notation (FEN), Universal Chess Interface (UCI), and Standard Algebraic Notation (SAN). The system will support exporting scenarios and moves to CSV or similar formats for archiving and version control. It will also facilitate data analysis principles based on chess data.

**AI Training & Competition Framework**: Provide methodologies and software for students to work with solution datasets to train Artificial Intelligence models. This includes enabling students to fine-tune AI training and measure performance. The software will culminate in a chess and AI competition where student-trained AIs compete, reflecting the collective intelligence of each group.

**Version Control Integration**: Utilize version control tools such as Git, specifically through platforms like GitHub, to ensure data continuity and seamless collaboration among partners for storing and sharing datasets of chess problem-solving challenges and solutions.

## Implementation plan

To achieve the ChessAIThon project's ambitious goals, a phased plan focusing on methodology development, platform creation, and student engagement is essential. This plan integrates the project's specific objectives and desired outcomes into actionable steps.

### Phase 1: Methodology and Educational Content Development (Months 1-6)

Objective: Develop the theoretical and practical framework for integrating chess, coding, and AI into VET curricula.

* Chapter 1: Coding Fundamentals through Chess: Develop curriculum and lesson plans for teaching logic, functions, and implementation of coding using chess logic and problem-solving strategies.
* Chapter 2: Transversal Skills Development: Create materials that highlight the synergy between coding and chess for fostering cognitive skills, creativity, lateral thinking, problem-solving, attention, concentration, perseverance, memory, spatial perception, time and space organization, and planning.
* Chapter 3: Chess Data Structures and File Formats: Design content explaining the representation of chess in various computer file formats, including PGN, JSON, CSV, FEN, UCI, and SAN.
* Chapter 4: Machine Learning and Data Analysis for Chess: Develop materials covering specific theoretical concepts and fundamental procedures of machine learning relevant to chess-based AI, focusing on movement solving.
* Chapter 5: Version Control and Dataset Sharing: Outline the use of version control tools like Git and platforms like GitHub for data continuity and collaboration in storing and sharing chess datasets.
* Lesson Plan Creation: Create a collection of selected chess problem-based scenarios supported by fully operative lesson plans, providing step-by-step guidance for use with students.
* Teacher Training Preparation: Prepare VET teachers to effectively teach coding, AI, and transversal skills using the developed methodology.

### Phase 2: Online Platform Development (Months 4-12)

Objective: Create a dynamic web platform and online database for real-life chess scenarios and AI training.

* Database Design: Design and implement a robust database to store chess scenarios (FEN format) and moves (SAN, UCI, or resulting FEN).
* Frontend Development: Develop an intuitive and visually appealing chessboard interface using the Chess.js library for displaying all possible legal moves.
* Scenario and Move Input: Implement functionality for users to add new cases and contribute solutions (next move only).
* Move Validation and Storage: Integrate logic to check the legality of moves and store valid moves in the database.
* Data Export Functionality: Develop export features to allow users to export scenario and move data to CSV or similar file formats for archiving and version control.
* AI Training Integration (Initial): Lay the groundwork for integrating the platform with AI training modules, allowing the aggregation of student and player proposals to train an AI.
* Version Control Integration (Platform Level): Ensure the platform design facilitates the use of version control programs to store chess scenarios.

### Phase 3: Student Engagement and AI Training (Months 9-18)

Objective: Engage students in practical "learning by doing" experiences, focusing on AI training and data analysis with chess scenarios.

* Student Onboarding and Training: Guide students in using the platform to solve chess scenarios and store optimal moves.
* Dataset Utilization: Engage students in using the generated datasets of chess scenarios and solutions to begin training their Artificial Intelligence tools.
* AI Fine-tuning and Performance Measurement: Enhance student skills in fine-tuning AI training and comparing/measuring enhanced performance in specific scenarios, particularly critical moments preceding checkmate in historically significant games.
* AI Evolution Observation: Guide students in observing the continuous evolution of AI scenario training to ensure proficiency in various chess situations.
* Preparation for Competition: Prepare students to train their AIs using automated tools or self-created ones, documenting moves in a dataset for the competition.

### Phase 4: Chess and AI Competition & Dissemination (Months 15-24)

Objective: Culminate the project with a practical competition and disseminate results and insights to the broader AI community.

* Competition Implementation: Host the transnational chess and AI competition, where different AIs shaped by student efforts compete against one another.
* Results Analysis and Documentation: Document the competition results and insights, contributing to the publicly available knowledge base.
* Knowledge Base Enrichment: Ensure the competition results and insights further enrich the broader AI community.
* Project Dissemination: Share the comprehensive learning resources, the online database, and the results of the competition with external actors, including students, companies, and governments.

This phased approach ensures a structured development process, maximizing collaboration and achieving the project's educational and technological objectives.

## Implementation memory

Based on the plan, we are going to focus on technical tasks in this memory. First part is to investigate about this topic, state of art and a find a viable solution with out hardware limitations. Here are some thoughts:


### AI Architecture

AI part will be heavily inspired in AlphaZero as a case on exit not hard to understand: https://arxiv.org/abs/1712.01815 

Here are some interesting links:  
* http://cs230.stanford.edu/projects_winter_2019/reports/15808948.pdf 
* https://ai.stackexchange.com/questions/27336/how-does-the-alpha-zeros-move-encoding-work

But this project is different. Were lack the hardware and time of Alphazero.

> The initial training of AlphaZero, where it learned chess from scratch by playing against itself, required immense computational resources. DeepMind used specialized hardware, specifically Google's Tensor Processing Units (TPUs), which are designed for machine learning workloads. For the famous matches against Stockfish 8, AlphaZero ran on a machine with four TPUs, providing a total processing power of about 180 TFLOPS. During the AlphaZero vs. Stockfish 8 matches, Stockfish was running on 44 CPU cores with a 32 GB hash size. Some sources indicate that AlphaZero had a significant hardware advantage (e.g., a 31x advantage, though the exact comparison can be debated due to different hardware types).

**So why we are learning AI if traditional algorithms are better?**

It's an excellent question that gets to the heart of why we're focusing on AI in the ChessAIThon project, especially when traditional algorithms like those in Stockfish have proven to be incredibly powerful. While traditional algorithms, particularly in domains with well-defined rules and searchable spaces like chess, can achieve exceptional performance through brute-force calculation and highly optimized heuristics, AI offers fundamentally different and complementary advantages.

In essence, while traditional algorithms are incredibly efficient for specific, well-defined problems, AI provides a powerful paradigm for building systems that can learn, adapt, and discover solutions in complex, dynamic, and often uncertain environments. Learning AI is not about replacing traditional algorithms, but about gaining a versatile toolset for a wider array of challenges and preparing for the future of technology.

**Differences with AlphaZero**

The primary difference between AlphaZero and Chessmarro, as outlined in this project, lies in their training methodology.

* AlphaZero and Leela Chess Zero learn by playing against themselves repeatedly, continually improving with each game.
* Chessmarro, in contrast, will be trained using chess games played by human participants (students) as examples. This approach aims to help the AI learn from actual human moves.

Both AI systems utilize Convolutional Neural Networks (CNNs) for chess move prediction, leveraging the grid-based structure of the chessboard for pattern recognition and efficient processing. While AlphaZero learns without human-made rules, Chessmarro's training dataset is based on student-documented chess moves.


## AI implementation

In this documentation there is a document about training ChessMarro, the first version of our AI. 

### Datasets

First steps are searching for real cases of chess moves. We found lots of them in Kaggle. Other users upload games from **Lichess**. We found 30000 games pre-mate and many complete games. All of them had to be transformed in our format. We used Kaggle first, but later we had to use Colab too: https://www.kaggle.com/code/xxjcaxx/convert-to-chessintion-format

The previous part is not necessary to repeat, we have more than enough here: https://www.kaggle.com/datasets/xxjcaxx/chess-games-in-fen-and-best-move

We can store data in JSON o Parquet. In JSON we decided to compress each of the 77 board in a int64 number. In Parquet is not necessary because it can compress better.

TODO: Decide best format and publish algorithms to convert. 

### Training

This is a Kaggle version of training:

https://www.kaggle.com/code/xxjcaxx/cnn-pytorch-chess-generic

This version uses only parquet files with generic games, not mates.

This version (obsolete) uses only mates and has 90% of precision https://www.kaggle.com/code/xxjcaxx/cnn-pytorch-chess-mates

The result CNN is not bad for the performance. Add more layers, or wider layers should increase precision, but it could be slower. We do lots of fine tunnings and this CNN and the result model is, for the moment, sufficient to go ahead.


### Deploying the model

We have a deployment in kaggle with MCTS: https://www.kaggle.com/code/xxjcaxx/launching-chessmarro

And a demo in kaggle to play with the model: https://www.kaggle.com/code/xxjcaxx/play-with-chessmarro

As we have lots of Kaggle and Colabs, we need to share the CNN and the model. We share it in Hugginface: https://huggingface.co/jocasal/chessmarrov1

We deployed with CPU in Huggingface: https://huggingface.co/spaces/jocasal/chessmarro/tree/main

And we have a Github repository: https://github.com/xxjcaxx/ai-libraries

Now, add this "AI Libraries" to the official repository.

All of these work were just drafts and trials. We need to centralize in:

* Official Github repository with:
  * AI model and CNN
  * Libraries in C++ for MCTS
  * Documentation (Docusarus in Github Pages)
  * Source Code of the frontend
  * Datasets
* Kaggle for:
  * Transform datasets to parquet
  * Train the CNN model (Could be Google Colab too)
  * Deploy the model
* HuggingFace for:
  * Deploy the model in CPU
  * Share the model to the community and our Kaggles
* Web App deployed with CI/CD from the github repository.


The workflow is:

Using Kaggle or Colab to improve model.
Upload new versions of the model to Github and HuggingFace
Using a link for this model in other Colab or Kaggle to test the model.
Deploy in HuggingFace for tests purposes.
Deploy in a local PC with GPU for the competition. 













https://github.com/carlos-paezf/Software_Construction?utm_source=chatgpt.com
