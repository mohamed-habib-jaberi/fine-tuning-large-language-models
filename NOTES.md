# Notes de préparation à un entretien — Fine-tuning de LLM

Ces notes expliquent le workflow de `llm_finetuning.ipynb`. Elles distinguent ce que le code prévoit et ce qu’il faudrait mesurer pour démontrer les résultats. Aucun entraînement supplémentaire n’a été exécuté pour écrire ce document.

## 1. Présenter le projet en une minute

« Ce projet pédagogique explore l’adaptation de Qwen2.5-1.5B-Instruct à des articles de presse en arabe. Les tâches sont l’extraction d’informations structurées et la traduction en français et en anglais. Le notebook génère des exemples avec GPT-4o-mini, les prépare au format Alpaca, puis configure un fine-tuning supervisé avec LoRA via LLaMA-Factory. Il présente aussi le chargement des adaptateurs, le service d’inférence avec vLLM et des tests de charge avec Locust. Pour valider les gains, je comparerais les modèles sur des articles indépendants avec des métriques de qualité et de performance. »

En entretien, préciser les étapes que l’on a personnellement exécutées. Ne pas annoncer un gain de précision ou de vitesse sans mesure vérifiée.

## 2. Comprendre le processus complet

```text
Articles arabes : news-sample.jsonl
        ↓ lecture du champ content
GPT-4o-mini : extraction + traduction anglais + traduction français
        ↓ réponses générées et parsées
sft.jsonl : story + task + output_scheme + response
        ↓ conversion en consigne / réponse attendue
Format Alpaca : system + instruction + input + output + history
        ↓ création de train.json et val.json
LLaMA-Factory : entraînement SFT avec LoRA
        ↓ sauvegarde des adaptateurs
Qwen de base + adaptateurs → génération Transformers ou serveur vLLM
        ↓
Comparaison des réponses et test de charge Locust
```

### Étape A — Charger les données

Un fichier JSONL contient un objet JSON par ligne. Le notebook lit `news-sample.jsonl`, ignore les lignes vides et transforme chaque ligne en dictionnaire Python avec `json.loads`. L’article se trouve dans `story['content']`.

`random.Random(101).shuffle(raw_data)` mélange les articles avec une graine fixe. Cela reproduit le mélange pour une même liste initiale ; cela ne garantit pas à lui seul un entraînement entièrement déterministe.

### Étape B — Définir ce que le modèle doit produire

Pydantic décrit les champs et contraintes attendus :

- `NewsDetails` : titre, mots-clés, résumé, catégorie et entités.
- `TranslatedStory` : titre traduit et contenu traduit.

`model_json_schema()` fournit une description JSON de cette structure. Elle est incluse dans la consigne pour guider le modèle.

**Point important :** fournir un schéma dans le prompt ne garantit pas une réponse conforme. Le notebook répare et parse le JSON avec `json_repair`, mais n’effectue pas une validation Pydantic systématique des réponses. Une amélioration serait `NewsDetails.model_validate(response_dict)` ou son équivalent pour la traduction. Même un JSON valide peut contenir des informations fausses.

### Étape C — Générer des exemples supervisés

GPT-4o-mini reçoit l’article, la tâche et le schéma. Sa réponse devient une cible d’apprentissage pour le petit modèle. C’est une distillation par exemples générés : on transmet un comportement via les réponses du modèle enseignant.

Il ne s’agit pas ici de comparer les probabilités internes des deux modèles pendant l’entraînement. Le notebook sauvegarde uniquement les réponses textuelles structurées.

Un article peut donner jusqu’à trois exemples : extraction, traduction anglaise et traduction française. Certaines réponses sont ignorées si la génération ne se termine pas normalement ou si le parsing échoue.

### Étape D — Comprendre les champs de sft.jsonl

| Champ | Origine | Rôle |
| --- | --- | --- |
| `id` | Compteur dans la boucle | Identifier un exemple |
| `story` | `story['content']` du dataset initial | Texte arabe |
| `task` | Consigne écrite dans le code | Action demandée |
| `output_scheme` | Schéma Pydantic sérialisé | Structure attendue |
| `response` | Réponse parsée de GPT-4o-mini | Cible d’apprentissage |

Dans la boucle suivante, `rec = json.loads(line.strip())` lit l’un de ces exemples. Ainsi, `rec['story']`, `rec['task']` et `rec['output_scheme']` viennent du fichier intermédiaire, pas d’une recherche automatique dans le texte.

### Étape E — Construire le format Alpaca

`finetuning_data.append({...})` ajoute un dictionnaire représentant un exemple d’entraînement.

| Champ Alpaca | Explication simple |
| --- | --- |
| `system` | Règles générales : suivre la tâche et fournir le JSON |
| `instruction` | Article + tâche + schéma + indication du début de réponse |
| `input` | Vide, puisque l’article est déjà dans `instruction` |
| `output` | Réponse attendue provenant de `rec['response']` |
| `history` | Vide : aucune conversation précédente |

`'\n'.join([...])` assemble des textes avec des retours à la ligne. `json.dumps` transforme un objet Python en texte JSON ; `ensure_ascii=False` conserve les caractères arabes, et `default=str` convertit en chaîne les objets autrement non sérialisables. Ce dernier réglage ne remplace pas une validation des données.

Les libellés `# Story:`, `# Task:` et `# Output Scheme:` sont des titres écrits par le développeur. Le modèle n’impose pas ces noms ; ils rendent la consigne lisible et cohérente.

**Exemple simplifié :**

```text
Consigne : Voici un article arabe. Extrais ses entités selon ce schéma.
Réponse attendue : {"story_entities": [...]}
```

Le notebook ouvre un bloc Markdown JSON à la fin de la consigne et en ajoute un autre au début de la réponse. Pour améliorer les données, harmoniser ce format et vérifier le texte final obtenu après application du template de conversation.

### Étape F — Séparer les données

Le notebook mélange les exemples, prend les 2 700 premiers pour l’entraînement et le reste pour la validation. Il faut vérifier la taille du dataset pour que la validation ne soit pas vide.

**Limite importante :** les différentes tâches d’un même article peuvent être réparties entre entraînement et validation. Le modèle a alors déjà vu le contenu de l’article : c’est une fuite de données.

La bonne amélioration consiste à séparer les articles sources en entraînement, validation et test **avant** de générer les différentes tâches. Dédupliquer aussi les articles similaires.

## 3. Les techniques à connaître

### Préentraînement, SFT et inférence

- **Préentraînement** : apprendre des régularités du langage sur de grandes quantités de textes. Ce projet utilise un modèle déjà préentraîné et déjà adapté aux instructions.
- **SFT, Supervised Fine-Tuning** : poursuivre l’entraînement sur des paires consigne / réponse attendue pour spécialiser son comportement.
- **Inférence** : utiliser le modèle entraîné pour produire une réponse, sans mise à jour de ses poids.

Un modèle causal prédit le prochain token à partir des tokens précédents. Pendant le SFT, la loss mesure à quel point les prédictions s’écartent des tokens attendus. Dans un entraînement sur réponses, les tokens du prompt sont généralement masqués dans le calcul de la loss ; vérifier les options et le prétraitement de la version de LLaMA-Factory utilisée.

### Tokenisation et chat template

Un token est une unité de texte, parfois un mot, un morceau de mot ou un signe. Le nombre de tokens varie selon la langue et le tokenizer.

Le tokenizer convertit le texte en identifiants numériques. `apply_chat_template` ajoute le format de conversation attendu par Qwen, notamment les rôles `system`, `user` et `assistant`. Employer le bon template évite un décalage entre entraînement et utilisation.

### LoRA et PEFT

PEFT désigne les méthodes qui entraînent seulement une petite partie des paramètres. LoRA est l’une de ces méthodes.

Au lieu de modifier toute une matrice de poids `W`, LoRA apprend une correction à travers deux petites matrices :

```text
W adapté = W de base + correction
correction = facteur d’échelle × B × A
```

Les poids de base restent gelés ; les matrices LoRA sont entraînées. Le rang `r` contrôle la taille de cette correction. Pour une matrice de dimensions `d_out × d_in`, LoRA ajoute environ `r × (d_in + d_out)` paramètres, plutôt que d’entraîner les `d_out × d_in` poids de cette matrice.

Le notebook utilise `lora_rank: 64` et `lora_target: all`, c’est-à-dire toutes les couches prises en charge comme cibles par l’outil, pas tous les paramètres du modèle.

**Pourquoi LoRA ?** Réduire les paramètres entraînables, les états de l’optimiseur et la taille des artefacts sauvegardés. Le modèle de base et les activations occupent toujours de la mémoire GPU.

**LoRA ou QLoRA ?** QLoRA combine des adaptateurs LoRA avec un modèle de base quantifié, souvent en 4 bits. Ici, la configuration montre LoRA avec BF16 ou FP16 selon le GPU ; l’import de `BitsAndBytesConfig` ne suffit pas à faire de ce projet un entraînement QLoRA.

### BF16 et quantification

BF16 est une précision numérique sur 16 bits, utile pour diminuer la mémoire par rapport au FP32 sur du matériel compatible. Ce n’est pas une quantification en 4 bits. Le serveur vLLM du notebook utilise `--dtype=half`, qui correspond au FP16.

### Hyperparamètres du notebook

| Paramètre | Valeur | Explication |
| --- | --- | --- |
| `cutoff_len` | 3500 | Limite de tokens utilisée pour préparer les exemples ; risque de troncature |
| Batch par GPU | 1 | Un exemple par microbatch |
| Accumulation | 4 | Cumuler les gradients sur quatre microbatches avant la mise à jour |
| Learning rate | `1e-4` | Taille des mises à jour des paramètres entraînés |
| Époques | 3 | Trois passages sur les données d’entraînement |
| Scheduler | Cosine | Réduire progressivement le learning rate |
| Warmup | 0,1 | Augmenter progressivement le learning rate sur la phase initiale |
| Évaluation | Toutes les 100 étapes | Suivre la validation pendant l’entraînement |
| Sauvegarde | Toutes les 500 étapes | Conserver des checkpoints |

Sur un GPU, le batch effectif est `1 × 4 = 4`. En entraînement distribué, multiplier aussi par le nombre de processus de données. L’accumulation économise la mémoire des activations par microbatch, mais n’offre pas nécessairement la même vitesse qu’un grand batch.

### Overfitting et validation

L’overfitting apparaît quand le modèle s’adapte trop aux exemples vus et généralise mal. Une loss d’entraînement qui diminue tandis que la loss de validation augmente est un signal possible.

Pour le limiter : données variées et propres, moins d’époques, sélection d’un checkpoint sur validation et ajustement des hyperparamètres. Le jeu de test sert à la mesure finale, pas au réglage répété du modèle.

## 4. Évaluation : comment prouver que cela marche ?

Le notebook présente des générations et des mesures de tokens ; cela ne démontre pas une amélioration générale de qualité.

Comparer le modèle de base et le modèle adapté sur les **mêmes articles de test**, avec des consignes et réglages de génération contrôlés. Si les prompts diffèrent, expliquer que la comparaison porte sur les workflows complets.

| Aspect | Mesure proposée |
| --- | --- |
| Structure | Taux de JSON parsable et conforme au schéma |
| Entités | Précision, rappel et F1, avec des annotations de référence |
| Catégorie | Accuracy ou macro-F1 si les classes sont déséquilibrées |
| Résumé | Fidélité factuelle, couverture et éventuellement ROUGE |
| Traduction | Relecture bilingue, chrF/BLEU et éventuellement COMET |
| Robustesse | Articles longs, catégories rares, noms propres et nombres |
| Performance | Latence, tokens/s, mémoire GPU, taux d’erreurs |

Précision : parmi les entités prédites, quelle proportion est correcte ? Rappel : parmi les entités attendues, quelle proportion a été retrouvée ? F1 combine les deux. Définir aussi ce qui compte comme une correspondance : texte exact, limites de l’entité et type.

La loss ne suffit pas : un modèle peut produire un JSON bien formé et inventer une entité. Les métriques automatiques de résumé ou de traduction doivent être complétées par une vérification du sens.

## 5. Déploiement, débit et coût

### Adaptateurs et vLLM

Après l’entraînement, on charge Qwen de base, puis les adaptateurs LoRA. Un adaptateur seul ne remplace pas le modèle complet.

vLLM sert le modèle derrière une API compatible avec les interfaces OpenAI. Il optimise notamment la gestion du cache KV et le traitement de requêtes concurrentes. Le cache KV conserve des informations d’attention des tokens déjà calculés pour éviter de refaire ces calculs à chaque nouveau token.

Le notebook active LoRA et expose l’adaptateur sous le nom `news-lora`. L’endpoint utilisé est `/v1/completions` : le prompt est déjà mis en forme avec le tokenizer.

### Locust

Locust simule des utilisateurs qui envoient des requêtes HTTP. L’exemple prévoit 20 utilisateurs, une montée en charge d’un utilisateur par seconde et une durée de 60 secondes. Les utilisateurs attendent entre 1 et 3 secondes entre les requêtes : cela ne signifie pas exactement 20 requêtes par seconde.

Mesurer les latences médiane, p95 et p99, les erreurs et le débit. Le p95 est la latence en dessous de laquelle se trouvent 95 % des requêtes. Distinguer le débit global du serveur et la vitesse d’une seule génération.

Les textes Faker testent la charge, pas la qualité sur de vrais articles. Un calcul avec un nombre de tokens divisé par 60 reste approximatif si l’on ne mesure pas la durée réelle et si l’on ne distingue pas entrée et sortie. Pour un benchmark solide : échauffement, matériel documenté, mêmes longueurs, même concurrence et comptage des requêtes échouées.

### Coût

```text
Coût API ≈ tokens d’entrée / 1 000 000 × tarif d’entrée
         + tokens de sortie / 1 000 000 × tarif de sortie
```

Les tarifs présents dans le notebook sont historiques : consulter les tarifs actuels. Son compteur n’inclut pas systématiquement toute la consommation des réponses rejetées ; il faut comptabiliser chaque appel facturé.

Le coût total comprend aussi le GPU pour l’entraînement, l’inférence, le stockage et la maintenance. Le fine-tuning ne rend pas automatiquement le projet moins cher : cela dépend du volume d’utilisation et de la qualité requise.

## 6. Limites concrètes et améliorations proposées

- **Réponses synthétiques imparfaites** : valider le schéma, relire un échantillon et contrôler les faits, noms et chiffres.
- **Fuite entre ensembles** : séparer les articles sources avant la génération des tâches.
- **Doublons** : les boucles ouvrent `sft.jsonl` en mode ajout ; une nouvelle exécution peut répéter les exemples.
- **Identifiants** : le compteur `ix` recommence à zéro dans les deux boucles ; employer des identifiants uniques et une clé article / tâche pour reprendre la génération.
- **Reproductibilité** : fixer les versions et le commit de LLaMA-Factory, conserver la configuration et les caractéristiques du GPU.
- **Prompt et réponse** : harmoniser les blocs JSON et vérifier que la troncature ne coupe pas la réponse cible.
- **Configuration** : adapter tous les chemins Drive et le dépôt Hugging Face avant l’exécution ; la configuration est centralisée dans `config.json`, et la publication Hugging Face est désactivée par défaut.
- **Inférence** : libérer la mémoire des modèles Transformers avant de lancer vLLM si nécessaire.
- **Filtrage de caractères** : le filtre de caractères chinois est une heuristique qui peut bloquer des sorties légitimes ; privilégier d’abord des données et consignes adaptées.

## 7. Questions d’entretien et réponses courtes

### Pourquoi utiliser un modèle déjà Instruct ?

Il sait déjà suivre des consignes. Le fine-tuning cherche à spécialiser ce comportement sur nos tâches et notre format, plutôt qu’à apprendre le langage depuis zéro.

### Pourquoi utiliser un modèle de 1,5 milliard de paramètres ?

C’est un candidat pour expérimenter avec moins de ressources qu’un grand modèle. Sa qualité en arabe et sa capacité sur les tâches doivent être mesurées ; sa petite taille ne garantit pas qu’il soit suffisant.

### Pourquoi ne pas utiliser uniquement GPT-4o-mini ?

C’est une baseline utile. Un modèle local adapté peut offrir davantage de contrôle et un autre compromis coût / latence. Il faut comparer les solutions avant de conclure.

### Fine-tuning, prompting ou RAG ?

Le prompting définit la tâche sans modifier les poids. Le fine-tuning adapte le comportement à partir d’exemples. Le RAG récupère des documents au moment de répondre pour fournir des informations externes ou récentes. Pour analyser un article déjà fourni dans le prompt, un RAG n’est pas nécessaire par défaut.

### La distillation garantit-elle un meilleur modèle ?

Non. Le modèle apprend aussi les erreurs de l’enseignant et peut perdre en généralisation. La qualité des exemples et un test indépendant sont essentiels.

### Pourquoi mettre la tâche et le schéma dans chaque exemple ?

Le modèle reçoit le contexte exact de l’action demandée. Cela permet de regrouper extraction et traduction dans un même dataset, avec des sorties différentes.

### Pourquoi laisser input et history vides ?

Tout le contenu nécessaire est dans `instruction`, et l’exemple ne contient aucun échange précédent. Ces champs servent à respecter le format choisi.

### Comment reconnaître une hallucination ?

Comparer la réponse à l’article et aux références. Une organisation ou un chiffre absent du texte peut être inventé, même si le JSON respecte parfaitement le schéma.

### Que faire si la mémoire GPU est insuffisante ?

Réduire la longueur des séquences ou le microbatch, activer des options comme le gradient checkpointing, puis envisager QLoRA ou un autre matériel. Mesurer l’effet sur la qualité et le temps de calcul.

### Que sauvegarde-t-on après LoRA ?

Les adaptateurs et leur configuration, ainsi que les éléments nécessaires pour les recharger. Conserver aussi l’identifiant et la version du modèle de base et du tokenizer.

### Quels résultats peut-on présenter aujourd’hui ?

Le workflow et la configuration présents dans le dépôt. Présenter des métriques seulement après les avoir reproduites et vérifiées ; les sorties d’un tutoriel ne sont pas automatiquement des résultats personnels.

## 8. Révision rapide avant l’entretien

Pouvoir expliquer sans lire :

1. L’objectif du projet et les tâches en arabe.
2. L’origine de chaque champ du dataset intermédiaire.
3. La différence entre SFT, LoRA, QLoRA et distillation.
4. Le rôle du tokenizer, du template et de la loss.
5. La différence entre validation, test et fuite de données.
6. Le rôle des adaptateurs, de vLLM et de Locust.
7. Les métriques à mesurer et les limites du notebook.

Phrase utile pour répondre à une question sur une étape non exécutée : « Le notebook prévoit cette étape. Pour la valider, voici le protocole que je mettrais en place… »

## Adaptation du dépôt

Les chemins des données et du modèle sont centralisés dans `config.json`. Le notebook enregistre automatiquement les datasets et génère le YAML. Les anciens comptes de suivi, les destinations de publication et les résultats écrits en dur ne servent plus à l’exécution. Les anciennes sorties ont été effacées pour permettre une nouvelle exécution traçable.

Ces corrections ne démontrent pas un gain de qualité : la compatibilité des dépendances et l’entraînement complet restent à vérifier sur GPU. Le découpage par article, la validation des réponses synthétiques et la reprise sans doublons restent des améliorations nécessaires.
