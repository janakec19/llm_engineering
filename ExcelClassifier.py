import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import requests

# CONFIG
print("Starting classification...")
# Load Excel file
input_file = 'D:\projects\python-project\AMData.xlsx'  # Replace with your actual file

#input_file = 'your_file.xlsx'
output_file = 'classified_with_labels.xlsx'
n_clusters = 5
model_name = 'mistral'  # or 'llama2', 'phi', etc.

# STEP 1: Load Excel and get descriptions
df = pd.read_excel(input_file)
descriptions = df['description'].astype(str).tolist()

# STEP 2: Embed descriptions using sentence-transformers
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(descriptions)

# STEP 3: Cluster using KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)
df['category_cluster'] = labels

# STEP 4: Function to name each cluster via Ollama
def name_cluster(examples, model_name='mistral'):
    joined_examples = "\n".join([f"- {e}" for e in examples])
    prompt = f"""You are a classification expert.

Given the following examples from one category, assign a short name (1-4 words) that best describes the pattern or topic they represent.

Examples:
{joined_examples}

Category name:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False}
    )

    if response.status_code == 200:
        return response.json()['response'].strip()
    else:
        return "Unnamed"

# STEP 5: Generate cluster names
cluster_names = {}
for cluster_id in sorted(df['category_cluster'].unique()):
    sample_texts = df[df['category_cluster'] == cluster_id]['description'].head(5).tolist()
    cluster_name = name_cluster(sample_texts, model_name)
    cluster_names[cluster_id] = cluster_name

# STEP 6: Map labels to dataframe
df['category_label'] = df['category_cluster'].map(cluster_names)

# STEP 7: Save to Excel
df.to_excel(output_file, index=False)
print(f"✔️ Cluster classification and naming complete.\nSaved to: {output_file}")
