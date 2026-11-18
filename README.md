# Fine-Tuning Large Language Models

Projet pédagogique de fine-tuning de **Qwen2.5-1.5B-Instruct** avec **LoRA** et **LLaMA-Factory**, à partir d’articles en arabe. Le workflow est présenté dans [`llm_finetuning.ipynb`](llm_finetuning.ipynb) et prévu pour **Google Colab avec un GPU NVIDIA**.

Pour préparer un entretien, consulter les [notes en français](NOTES.md) : explication du processus, techniques et questions pratiques.

## Objectifs

- Extraire un titre, des mots-clés, un résumé, une catégorie et des entités sous forme de JSON.
- Traduire les articles arabes vers l’anglais et le français.
- Générer des exemples supervisés avec `gpt-4o-mini` (distillation), puis entraîner des adaptateurs LoRA.
- Comparer les réponses du modèle de base et du modèle adapté sur des exemples.
- Servir le modèle avec vLLM et mesurer la charge avec Locust.

## Dataset

Les données sont disponibles dans ce [dossier Google Drive](https://drive.google.com/drive/folders/1Im6m8oW5vzyVNKr_Ur4Xjd9YMO4iCKD6). L’accès dépend des permissions de partage du dossier. Les fichiers Drive n’ont pas été inspectés lors de la rédaction de ce README ; les noms ci-dessous correspondent aux fichiers attendus par le notebook.

Copier les données dans son propre Drive, puis adapter `data_dir`. Monter Drive ne copie pas automatiquement le dossier partagé.

```text
llm-finetuning/
├── datasets/
│   ├── news-sample.jsonl
│   ├── sft.jsonl                         # généré par distillation, ou fourni
│   └── llamafactory-finetune-data/
│       ├── train.json                    # généré par le notebook
│       └── val.json                      # généré par le notebook
└── models/                               # adaptateurs et checkpoints générés
```

`news-sample.jsonl` doit contenir un objet JSON par ligne avec un champ `content` contenant le texte de l’article :

```json
{"content": "نص المقال باللغة العربية..."}
```

Les exemples sont convertis au format Alpaca (`system`, `instruction`, `input`, `output`, `history`). Le découpage actuel utilise les 2 700 premiers exemples mélangés pour l’entraînement et le reste pour la validation. Adapter cette valeur à la taille réelle du dataset et vérifier que la validation n’est pas vide. Pour une évaluation fiable, séparer les articles sources avant de générer leurs différentes tâches afin d’éviter de retrouver le même article dans les deux ensembles.

## Sites et outils utilisés

| Service / outil | Utilisation | Lien |
| --- | --- | --- |
| Google Colab | Exécution du notebook sur GPU | [colab.research.google.com](https://colab.research.google.com/) |
| Google Drive | Données et checkpoints | [drive.google.com](https://drive.google.com/) |
| Hugging Face | Téléchargement du modèle et hébergement optionnel des adaptateurs | [huggingface.co](https://huggingface.co/) |
| Qwen2.5-1.5B-Instruct | Modèle de base | [Fiche du modèle](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| LLaMA-Factory | Fine-tuning supervisé avec LoRA | [Dépôt GitHub](https://github.com/hiyouga/LLaMA-Factory) |
| Transformers | Chargement du modèle et génération | [Documentation](https://huggingface.co/docs/transformers/) |
| OpenAI API | Génération des réponses supervisées et comparaison | [platform.openai.com](https://platform.openai.com/) |
| Weights & Biases | Suivi des expériences et des pertes | [wandb.ai](https://wandb.ai/) |
| vLLM | Serveur d’inférence compatible avec l’API OpenAI | [Documentation](https://docs.vllm.ai/) |
| Pydantic | Définition des schémas JSON attendus | [Documentation](https://docs.pydantic.dev/) |
| Locust | Test de charge du serveur | [locust.io](https://locust.io/) |

## Démarrage dans Colab

1. Ouvrir `llm_finetuning.ipynb` dans Colab et sélectionner un environnement GPU.
2. Monter Google Drive avec la cellule prévue à cet effet.
3. Exécuter les cellules d’installation. Le notebook fixe notamment `transformers==4.48.3`, `datasets==3.2.0`, `optimum==1.24.0`, `openai==1.61.0`, `json-repair==0.29.1`, `faker==35.2.0` et `vllm==0.7.2`.
4. Ajouter dans les secrets Colab, avec l’accès au notebook activé :
   - `huggingface` : token Hugging Face ; droit d’écriture si publication des checkpoints.
   - `wandb` : clé Weights & Biases.
   - `openai-colab` : clé OpenAI pour les cellules de comparaison et de distillation.
5. Importer `config.json` dans Colab, puis adapter ses paramètres. La cellule de configuration charge ce fichier (ou le chemin indiqué par `PROJECT_CONFIG`). Exemple de valeurs :

   ```json
   {
   "data_dir": "/content/drive/MyDrive/llm-finetuning",
   "base_model_id": "Qwen/Qwen2.5-1.5B-Instruct"
   }
   ```

6. Exécuter les exemples de génération, puis les cellules de distillation si `sft.jsonl` n’est pas déjà disponible. Ces cellules effectuent des appels API facturés ; vérifier les tarifs actuels, les montants du notebook étant des valeurs historiques. Elles ajoutent les réponses à `sft.jsonl` : les relancer sur les mêmes données peut créer des doublons.
7. Convertir les exemples et créer `train.json` et `val.json`.
8. Exécuter la cellule qui enregistre automatiquement les deux datasets dans `dataset_info.json`.
9. Adapter la configuration YAML, puis lancer l’entraînement.
10. Charger les adaptateurs pour comparer les sorties, puis exécuter les sections vLLM et Locust si nécessaire.

## Configuration de l’entraînement

| Paramètre | Valeur dans le notebook |
| --- | --- |
| Méthode | SFT + LoRA, toutes les couches cibles |
| Rang LoRA | 64 |
| Longueur maximale | 3 500 tokens |
| Batch par GPU | 1 |
| Accumulation des gradients | 4 |
| Taux d’apprentissage | `1e-4` |
| Époques | 3 |
| Scheduler / warmup | Cosine / 0,1 |
| Précision | BF16 si compatible, sinon FP16 sur GPU |
| Évaluation / sauvegarde | Toutes les 100 / 500 étapes |

La configuration est écrite dans `examples/train_lora/news_finetune.yaml` du clone LLaMA-Factory :

```bash
cd /content/LLaMA-Factory
llamafactory-cli train examples/train_lora/news_finetune.yaml
```

## Configuration et limites

- `config.json` centralise le modèle, les données, la taille d’entraînement, le chemin LLaMA-Factory et le nom de l’expérience. Le clone est installé dans `/content/LLaMA-Factory` ; changer également la cellule d’installation si un autre emplacement est utilisé.
- La publication Hugging Face est désactivée par défaut. Pour l’activer, renseigner `hub_model_id` avec son propre espace de noms et utiliser un token autorisé à écrire.
- Le YAML est généré à partir de cette configuration ; la précision choisit BF16 sur les GPU compatibles, sinon FP16 sur GPU. Un GPU NVIDIA reste nécessaire pour le workflow d’entraînement et vLLM prévu ici.
- Importer `config.json` dans Colab séparément du notebook. Le notebook ne télécharge pas automatiquement les fichiers du dépôt.
- Les dépendances historiques et la branche courante de LLaMA-Factory n’ont pas été validées ensemble. Fixer le commit et les versions après une exécution réussie.
- Les réponses générées sont parsées, mais leur conformité Pydantic et leur exactitude factuelle doivent encore être contrôlées.
- Le découpage actuel est effectué par exemple. Séparer les articles sources avant génération pour éviter les fuites entre entraînement et validation.
- Les boucles de distillation ajoutent les réponses au fichier ; prévoir déduplication et reprise avant de les relancer.
- Libérer la mémoire Transformers avant de lancer vLLM si nécessaire. Les métriques du tutoriel ont été retirées : mesurer qualité, latence et débit sur son propre environnement.

Voir [NOTES.md](NOTES.md) pour les techniques et améliorations.

## Structure du dépôt

```text
.
├── README.md
├── NOTES.md
├── config.json
├── llm_finetuning.ipynb
└── .gitignore
```

Les données, clés API, adaptateurs et checkpoints restent hors du dépôt Git.
