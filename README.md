# Transfer Learning for NLP with TensorFlow Hub
https://github.com/richapatel93/Transfer_Learning_NLP_TFHub/blob/main/tensorflow.png
!  
*Banner placeholder—replace with a screenshot of your TensorBoard or Task 8 plots for visual impact!*

## Project Goal
The goal of this project is to leverage **transfer learning** in natural language processing (NLP) to classify insincere questions from the Quora Insincere Questions dataset. By utilizing pre-trained embeddings from TensorFlow Hub, I aimed to build, compare, fine-tune, and visualize multiple text classification models, achieving high accuracy while addressing real-world challenges like data imbalance. This project showcases my ability to handle complex NLP tasks, optimize model performance, and present results professionally—skills essential for data science and AI roles.

## Project Description
This project implements a full NLP pipeline using TensorFlow and TensorFlow Hub to classify questions as sincere (0) or insincere (1). The Quora dataset, with over 1.3 million questions, is heavily imbalanced—most questions are sincere, making insincere detection challenging. I trained three models with pre-trained embeddings, compared their performance, fine-tuned the best one, and visualized results with Matplotlib and TensorBoard, demonstrating a robust approach to binary text classification.

### Key Achievements
- Achieved a peak validation accuracy of **95.94%** using transfer learning with `universal-sentence-encoder-large/5`.
- Successfully trained and compared three models using embeddings of varying complexity.
- Overcame runtime instability and data imbalance challenges in Google Colab.
- Produced publication-quality visualizations for model comparison and training insights.

## What We Did
1. **Setup Environment**: Configured TensorFlow 2.18.0 and TensorFlow Hub 0.16.1 in Google Colab with GPU acceleration.
2. **Data Preparation**: Loaded the Quora dataset, which was initially biased toward sincere questions (class 0 dominant). Used **stratified sampling** (`stratify=y` in `train_test_split`) to ensure balanced class proportions in training (1%) and validation (0.1%) sets, preserving the original distribution for representative evaluation.
3. **Model Building**: Designed a custom Keras layer to integrate TensorFlow Hub embeddings into a Sequential model, adding dense layers for classification.
4. **Transfer Learning**: Trained three models using pre-trained embeddings:
   - **Model_1**: `nnlm-en-dim50/2` (50-dim, lightweight).
   - **Model_2**: `universal-sentence-encoder/4` (512-dim, balanced).
   - **Model_3**: `universal-sentence-encoder-large/5` (512-dim, high-capacity).
5. **Model Comparison**: Evaluated models over 5 epochs, achieving validation accuracies of 93.80%, 94.87%, and 95.94%.
6. **Fine-Tuning**: Fine-tuned Model_3 with a low learning rate (1e-5) over 3 epochs, reaching a validation accuracy of 93.80%—a slight dip from transfer learning, suggesting the pre-trained embedding was already optimal for this task.
7. **Visualization**: 
   - Plotted accuracy and loss curves for all models using Matplotlib (Task 8).
   - Implemented TensorBoard for interactive visualization of fine-tuned Model_3 (Task 10).
8. **Optimization**: Addressed runtime disconnects by adjusting batch sizes, epochs, and memory management.

## Technologies Used
- **Python 3.11**: Core programming language.
- **TensorFlow 2.18.0**: Deep learning framework for model building and training.
- **TensorFlow Hub 0.16.1**: Source of pre-trained NLP embeddings.
- **Pandas & NumPy**: Data preprocessing and manipulation.
- **Scikit-Learn**: Stratified sampling for balanced data splits.
- **Matplotlib**: Static visualization of model performance.
- **TensorBoard**: Interactive visualization of training metrics and model graphs.
- **Google Colab**: Cloud-based environment with GPU support.

## Skills Demonstrated
- **Machine Learning**: Transfer learning, fine-tuning, hyperparameter optimization.
- **NLP**: Text classification with pre-trained embeddings, handling large datasets.
- **Deep Learning**: Custom Keras layer design, model architecture engineering.
- **Data Science**: Data preprocessing, stratified sampling, performance evaluation, visualization.
- **Problem-Solving**: Debugging runtime errors, managing imbalanced data, optimizing resources.
- **Communication**: Clear documentation and visual storytelling for technical audiences.
- **Software Engineering**: Modular code structure, reproducibility via GitHub.

## Key Challenges Overcome
1. **Data Imbalance**: The dataset was heavily skewed toward sincere questions. I used stratified sampling to maintain class proportions in train/validation splits, ensuring fair model evaluation despite the bias.
2. **Runtime Disconnects**: Training large models (e.g., Model_3 took ~250s/epoch) caused Colab runtime crashes. Mitigated by:
   - Reducing batch sizes (32 → 16) and epochs (5 → 3) for stability.
   - Clearing memory with `tf.keras.backend.clear_session()` between models.
   - Leveraging GPU acceleration for faster computation.
3. **TensorFlow Hub Integration**: Initial `KerasTensor` errors with `hub.KerasLayer` required a custom `HubLayer` wrapper to ensure compatibility with Sequential API.
4. **Fine-Tuning Balance**: Fine-tuning Model_3 resulted in a validation accuracy drop (95.94% → 93.80%), highlighting the need for further hyperparameter tuning or data balancing to unlock gains.

## Results
- **Model_1 (NNLM)**: Final Val Accuracy: 93.80% (lightweight baseline).
- **Model_2 (USE)**: Final Val Accuracy: 94.87% (balanced performance).
- **Model_3 (USE-Large)**: Final Val Accuracy: 95.94% (best transfer learning result).
- **Fine-Tuned Model_3**: Final Val Accuracy: 93.80% (fine-tuning outcome, analyzed below).

### Fine-Tuning Analysis
Fine-tuning Model_3 with `trainable=True` and a learning rate of 1e-5 over 3 epochs yielded a validation accuracy of 93.80%, lower than the transfer learning result (95.94%). Loss also increased (0.1026 → 0.5738), suggesting the pre-trained embedding was already well-aligned with the task, and fine-tuning may have over-adjusted it. This insight underscores the importance of careful hyperparameter tuning and data balancing in fine-tuning workflows.

### Screenshots
- **Task 7 Output**: Model accuracies  
  ![Task 7 Output](screenshots/https://github.com/richapatel93/Transfer_Learning_NLP_TFHub/blob/main/Accuracy%20and%20loss%20curves.png
- **Task 8 Plots**: Accuracy and loss curves  
  ![Task 8 Plots](screenshots/task8_plots.png
- **Task 9 Fine-Tuning**: Fine-tuned Model_3 results  
  ![Task 9 Output](screenshots/task9_output.png)
- **Task 10 TensorBoard**: Interactive visualizations  
   https://github.com/richapatel93/Transfer_Learning_NLP_TFHub/blob/main/tensorflow.png

*Replace `screenshots/*.png` with paths to your actual screenshots once uploaded to GitHub.*

## How to Run
1. Clone this repository: `git clone https://github.com/yourusername/transfer-learning-nlp.git`
2. Open `Transfer_Learning_NLP_TFHub.ipynb` in Google Colab.
3. Ensure GPU runtime is enabled (`Runtime > Change runtime type > GPU`).
4. Run all cells to reproduce the results.
5. Launch TensorBoard in Colab to explore interactive metrics.

## Future Improvements
- **Advanced Sampling Techniques**: Explore methods like **SMOTE** (Synthetic Minority Oversampling Technique) or **oversampling** the minority class (insincere questions) in the target dataset to address imbalance and potentially enhance fine-tuning performance.
- Experiment with additional embeddings (e.g., BERT) for higher accuracy.
- Implement cross-validation for more robust evaluation.
- Add custom metrics (e.g., F1-score) to better evaluate performance on imbalanced data.
- Optimize fine-tuning hyperparameters (e.g., learning rate, epochs) with grid search to improve results.



---

