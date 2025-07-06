import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Flatten, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import math
from urllib.parse import urlparse

class FeatureExtractor:
    def __init__(self, url=""):
        self.url = url
        parsed = urlparse(url)
        self.domain = parsed.netloc or url.split('//')[-1].split('/')[0]
        self.path = parsed.path

    # URL 熵
    def url_entropy(self):
        url_trimmed = self.url.strip()
        if len(url_trimmed) == 0:
            return 0
        entropy_distribution = [float(url_trimmed.count(c)) / len(url_trimmed) for c in dict.fromkeys(list(url_trimmed))]
        return -sum([e * math.log(e, 2) for e in entropy_distribution if e > 0])

    def digits_num(self):
        return sum(c.isdigit() for c in self.url)

    def url_length(self):
        return len(self.url)

    def param_nums(self):
        return self.url.count('&')

    def fragments_num(self):
        return self.url.count('#')

    def url_depth(self):
        return len([p for p in self.url.split('/') if p])  # 排除空项

    def domain_length(self):
        return len(self.domain)

    def subdomain_length(self):
        parts = self.domain.split('.')
        return len(parts[0]) if len(parts) > 2 else 0  # 只有子域名存在才统计长度

    def has_http(self):
        return 1 if 'http://' in self.url else 0

    def has_https(self):
        return 1 if 'https://' in self.url else 0

    def is_ip(self):
        parts = self.domain.split('.')
        return int(len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts))

    def dom_ext(self):
        parts = self.domain.split('.')
        return parts[-1] if len(parts) > 1 else ''

    # 特殊字符计数
    def count_special_char(self, char):
        return self.url.count(char)

    def run(self):
        return {
            "url_length": self.url_length(),
            "has_ip": self.is_ip(),
            "digits": self.digits_num(),
            "param_nums": self.param_nums(),
            "fragments_num": self.fragments_num(),
            "url_depth": self.url_depth(),
            "domain_length": self.domain_length(),
            "subdomain_length": self.subdomain_length(),
            "has_http": self.has_http(),
            "has_https": self.has_https(),
            "dom_ext": self.dom_ext(),
            "url_entropy": self.url_entropy(),
            "num_dots": self.count_special_char('.'),
            "num_hyphens": self.count_special_char('-'),
            "num_slash": self.count_special_char('/'),
            "num_questionmark": self.count_special_char('?'),
            "num_at": self.count_special_char('@'),
            "num_and": self.count_special_char('&'),
            "num_tilde": self.count_special_char('~'),
            "num_comma": self.count_special_char(','),
            "num_plus": self.count_special_char('+'),
            "num_asterisk": self.count_special_char('*'),
            "num_colon": self.count_special_char(':'),
            "num_semicolon": self.count_special_char(';')
        }

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# 自定义的特征提取器（需先定义 FeatureExtractor 类）
# from your_feature_extractor_file import FeatureExtractor

# -------------------------
# 2. 读取数据 + 特征提取
# -------------------------
df = pd.read_csv("H:/s_and_r/.venv/ai_training_data(2).csv")

# 结构化特征提取
features = [FeatureExtractor(url).run() for url in tqdm(df['URL'])]
features_df = pd.DataFrame(features)

# URL Token
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['URL'])
sequences = tokenizer.texts_to_sequences(df['URL'])
max_sequence_length = 128
X_seq = pad_sequences(sequences, maxlen=max_sequence_length)

# 标签编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Label'])  # 0 = good, 1 = bad

# 新的结构化特征字段列表
structured_columns = [
    "url_length", "has_ip", "digits", "param_nums", "fragments_num", "url_depth",
    "domain_length", "subdomain_length", "has_http", "has_https", "url_entropy",
    "num_dots", "num_hyphens", "num_slash", "num_questionmark", "num_at", "num_and",
    "num_tilde", "num_comma", "num_plus", "num_asterisk", "num_colon", "num_semicolon"
]

# 结构化特征转为 NumPy 数组
X_struct = features_df[structured_columns].astype(np.float32).values

# 数据划分：训练集 / 测试集
X_seq_train, X_seq_test, X_struct_train, X_struct_test, y_train, y_test = train_test_split(
    X_seq, X_struct, y, test_size=0.2, random_state=42
)


# 显示前几行特征数据（默认显示全部列）
pd.set_option('display.max_columns', None)
print("提取后的特征数据（前5行）：")
print(features_df.head())

# 导出为 CSV 文件
output_path = "H:/s_and_r/url_extracted_features.csv"
features_df.to_csv(output_path, index=False)
print(f"特征数据已成功导出到: {output_path}")
features_df["Label"] = df["Label"]
features_df.to_csv("H:/s_and_r/.venv/url_extracted_features_with_label.csv", index=False)

# -------------------------
# 3. 构建融合模型（LSTM + 手工特征）
# -------------------------
# 序列输入分支
input_seq = Input(shape=(max_sequence_length,), name='url_seq')
embedding = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64)(input_seq)
lstm = LSTM(64)(embedding)

# 手工特征输入分支
# input_struct = Input(shape=(9,), name='manual_features')
input_struct = Input(shape=(23,), name='manual_features')


# 融合 + 分类
concat = Concatenate()([lstm, input_struct])
dense = Dense(64, activation='relu')(concat)
dropout = Dropout(0.5)(dense)
output = Dense(1, activation='sigmoid')(dropout)

# 构建模型
model = Model(inputs=[input_seq, input_struct], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit([X_seq_train, X_struct_train], y_train, epochs=10, batch_size=32,
          validation_split=0.2)



from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 模型评估
y_pred_prob = model.predict([X_seq_test, X_struct_test])
y_pred = (y_pred_prob > 0.5).astype("int32")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=9))

# ROC-AUC
auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {auc:.4f}")

# ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()