# Fine-Tuning Large Language Models

Projet pédagogique de fine-tuning de **Qwen2.5-1.5B-Instruct** avec **LoRA** et **LLaMA-Factory**, à partir d’articles en arabe. Le workflow est présenté dans [`llm_finetuning.ipynb`](llm_finetuning.ipynb) et prévu pour **Google Colab avec un GPU NVIDIA**.

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
5. Adapter le chemin des données :

   ```python
   data_dir = "/gdrive/MyDrive/llm-finetuning"
   base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
   ```

6. Exécuter les exemples de génération, puis les cellules de distillation si `sft.jsonl` n’est pas déjà disponible. Ces cellules effectuent des appels API facturés ; vérifier les tarifs actuels, les montants du notebook étant des valeurs historiques. Elles ajoutent les réponses à `sft.jsonl` : les relancer sur les mêmes données peut créer des doublons.
7. Convertir les exemples et créer `train.json` et `val.json`.
8. Enregistrer `news_finetune_train` et `news_finetune_val` dans `/content/LLaMA-Factory/data/dataset_info.json`, avec les chemins correspondant à son Drive.
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
| Précision | BF16 |
| Évaluation / sauvegarde | Toutes les 100 / 500 étapes |

La configuration est écrite dans `examples/train_lora/news_finetune.yaml` du clone LLaMA-Factory :

```bash
cd /content/LLaMA-Factory
llamafactory-cli train examples/train_lora/news_finetune.yaml
```

## Points à adapter avant l’exécution

- La cellule « Configure LLaMA-Factory » contient un extrait JSON dans une cellule de code : il faut le reporter dans `dataset_info.json`, et non l’exécuter comme du Python.
- Les chemins de `dataset_info.json`, du YAML, du chargement des adaptateurs et du serveur vLLM sont écrits en dur. Les modifier en plus de `data_dir`.
- Le YAML contient `push_to_hub: true` et `export_hub_model_id: "bakrianoo/news-analyzer"`. Remplacer ce dépôt par le sien ou désactiver la publication avec `push_to_hub: false`.
- BF16 nécessite un GPU compatible. Adapter la précision et la taille des séquences à son matériel.
- LLaMA-Factory est cloné depuis sa branche courante, alors que plusieurs dépendances sont anciennes. La compatibilité de cet ensemble n’a pas été validée ; pour rendre une exécution reproductible, conserver le commit et les versions utilisés lors d’un entraînement réussi.
- Installer Locust avant les tests de charge (`pip install locust`). La section vLLM nécessite assez de mémoire GPU disponible, notamment après les générations Transformers.
- Les exemples d’évaluation et de débit du notebook ne constituent pas une mesure validée de qualité sur un jeu de test indépendant. Aucune exécution d’entraînement n’a été réalisée pour cette préparation du dépôt.

## Structure du dépôt

```text
.
├── README.md
├── llm_finetuning.ipynb
└── .gitignore
```

Les données, clés API, adaptateurs et checkpoints restent hors du dépôt Git.
