# Technical Memory

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

> The best dataset prepared to train is: https://www.kaggle.com/datasets/xxjcaxx/chessmarro-dataset/data 



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

#### Multi process

In this repository will be a docker that implements mcts in multi process to improve cpu and model performance. 


* **Batching mechanism**: Instead of letting every client process call the model individually, requests are accumulated into batches. This ensures efficient GPU use by processing several positions at once.

* **Dedicated worker process**: A separate worker process owns the model and runs inference. This avoids re-initializing CUDA in forked processes and keeps all predictions centralized.

* **Queues for communication**:

  * An **input queue** is used to send batches of board tensors and their associated task IDs to the worker.
  * An **output queue** carries back the predictions, paired with their task IDs.

* **Task identifiers**: Each client request is assigned a unique ID so predictions can be matched back to the correct requester, even if requests from multiple processes are mixed in a single batch.

* **Shared response channels**: Each client process provides a response queue, created via a `multiprocessing.Manager`, that is stored in a shared dictionary (`pending`). This makes it possible for the dispatcher to place results into the right client’s queue, across process boundaries.

* **Dispatcher loop**: A separate loop continuously monitors the output queue, extracts predictions, looks up the corresponding response queue in the shared dictionary, and forwards the result.

* **Shared state via Manager**: Critical data structures such as `pending` and the client response queues are created through a `Manager` so they are visible and usable across all processes, avoiding the issue of processes only having local copies.

This design allows multiple processes to submit requests concurrently, have them batched efficiently for inference, and still receive their individual predictions reliably.

Here’s the **step-by-step lifecycle flow** of how a client request travels through the system until the prediction is delivered back:

1. **Client creates a request**

   * A board tensor is prepared.
   * A new response queue (via `Manager`) is created for this request.
   * The request is assigned a unique task ID.
   * The request (tensor + task ID) is added to the batcher, and the response queue is registered in the shared `pending` dictionary under that task ID.

2. **Batching**

   * The batcher collects incoming requests until the batch size is reached (or until explicitly flushed).
   * Once ready, the batcher sends the accumulated tensors and their task IDs as a single item to the **input queue**.

3. **Worker process**

   * The worker continuously listens to the input queue.
   * When a batch arrives, it stacks the tensors into one batch tensor and runs the model on the GPU.
   * The predictions are generated and paired with the task IDs from the batch.
   * The list of `(task_id, prediction)` results is placed onto the **output queue**.

4. **Dispatcher loop**

   * A dispatcher process listens to the output queue.
   * For each `(task_id, prediction)` in the results:

     * It looks up the correct response queue in the shared `pending` dictionary.
     * It places the prediction into that response queue.
     * It removes the entry from `pending` once delivered.

5. **Client receives result**

   * The client waits on its response queue (`q.get()`).
   * As soon as the dispatcher places the prediction there, the client unblocks and receives the move.

6. **Cleanup**

   * When shutting down, the batcher flushes any remaining requests, sends a termination signal to the worker, and joins all processes cleanly.

This flow guarantees that:

* Multiple clients across processes can safely submit requests.
* The model runs only once per batch in a controlled worker process.
* Each client receives the correct prediction asynchronously, even if batches mix requests from different processes.


## Discarted CNN

```python

class ChessNetPV(nn.Module):
    def __init__(self):
        super(ChessNetPV, self).__init__()

        # Model parameters
        bit_layers = 77
        in_channels = bit_layers
        base_channels = 128  # Base number of channels  # Increase!!
        kernel_size = 3
        padding = kernel_size // 2
        lineal_channels = 1024

        # First convolution layer (no residual needed)
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # Second convolution with residual
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.res_conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=1)  # 1x1 conv to match channels

        # Third convolution with residual
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        self.res_conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=1)

        # Fourth convolution with residual
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(base_channels * 8)
        self.res_conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(base_channels * 8 * 8 * 8, lineal_channels)  # Retain spatial info
        self.drop1 = nn.Dropout(p=0.4)  # Lower dropout for better accuracy

        self.fc2 = nn.Linear(lineal_channels, lineal_channels)
        self.drop2 = nn.Dropout(p=0.4)

        # Política: Salida de 4096 movimientos
        self.fcf = nn.Linear(lineal_channels, 4096)
        
        # Valor: Salida de 1 escalar (Evaluación de la posición)
        self.fc_value_1 = nn.Linear(lineal_channels, 256)
        self.fc_value_2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh() # Para rango [-1, 1]

    def forward(self, x):
        # First convolution (no residual)
        x = F.relu(self.bn1(self.conv1(x)))

        # Second layer with residual
        res = self.res_conv2(x)
        x = F.relu(self.bn2(self.conv2(x))) + res

        # Third layer with residual
        res = self.res_conv3(x)
        x = F.relu(self.bn3(self.conv3(x))) + res

        # Fourth layer with residual
        res = self.res_conv4(x)
        x = F.relu(self.bn4(self.conv4(x))) + res

        # Flatten while keeping spatial information
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        # Policy
        policy = self.fcf(x)

        # Value
        value = F.relu(self.fc_value_1(x))
        value = self.tanh(self.fc_value_2(value))
        

        return policy, value



class ResBlock(nn.Module):
    """
    Standard Residual Block with 2 convolutions.
    Keeps information flow stable allowing for deeper networks.
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ChessNetPV2(nn.Module):
    def __init__(self):
        super(ChessNetPV2, self).__init__()

        # --- Architecture Configuration ---
        # Instead of doubling channels rapidly (which explodes parameters),
        # we use a constant channel depth with more layers (ResNet Tower).
        # This is the AlphaZero/Leela approach.
        self.input_channels = 77
        self.tower_channels = 256  # High capacity, constant depth
        self.num_res_blocks = 6    # Can be increased (e.g., 10, 20) for stronger play without massive param growth
        
        # --- Input Stem ---
        self.conv_input = nn.Conv2d(self.input_channels, self.tower_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(self.tower_channels)

        # --- Residual Tower ---
        self.res_tower = nn.Sequential(
            *[ResBlock(self.tower_channels) for _ in range(self.num_res_blocks)]
        )

        # --- Policy Head ---
        # We reduce channels to 32 before flattening. 
        # Old method: 1024 channels * 64 squares = 65,536 inputs to FC (Too big!)
        # New method: 32 channels * 64 squares = 2,048 inputs to FC (Efficient!)
        self.policy_conv = nn.Conv2d(self.tower_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096) # Output matches original requirement

        # --- Value Head ---
        # Reduces to 16 channels, then small dense layers.
        self.value_conv = nn.Conv2d(self.tower_channels, 16, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # 1. Stem
        x = F.relu(self.bn_input(self.conv_input(x)))

        # 2. Residual Tower
        x = self.res_tower(x)

        # 3. Policy Head
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1) # Flatten
        policy = self.policy_fc(p)
        # Note: LogSoftmax or Softmax is usually applied in the loss function, 
        # but raw logits are standard output for the model class.

        # 4. Value Head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1) # Flatten
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
```


# Links:

* Convert from FEN to 77x8x8 format: https://www.kaggle.com/code/xxjcaxx/convert-to-chessintion-format?scriptVersionId=259827731
* Dataset: https://www.kaggle.com/datasets/xxjcaxx/chessmarro-dataset/data
* Train: https://www.kaggle.com/code/xxjcaxx/cnn-pytorch-chess-generic 
* Deploy example: https://www.kaggle.com/code/xxjcaxx/launching-chessmarro
* Github: https://github.com/xxjcaxx/ChessAIThon
