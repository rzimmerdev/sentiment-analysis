# Redes Neurais e Aprendizado Profundo - Análise de Sentimento

## Grupo:

| Nome           | nUSP     |
|----------------|----------|
| Rafael Zimmer  | 12542612 |
| Murilo Soave   | 10688813 |
| Fernando Cesar | 10260559 |

Tarefa: Classificação
Dados: "Sentiment Analysis for Financial News"

## Descrição da Tarefa

A tarefa consiste em realizar a análise de sentimentos em um conjunto de dados de frases financeiras.
O objetivo é classificar cada frase como positiva, negativa ou neutra
(3 classes, ou seja, multi-class single-label) em relação ao seu conteúdo emocional.
Para isso, utilizamos três abordagens diferentes de modelagem de forma comparativa:

- uma abordagem baseline com Bag of Words;
- uma abordagem state-of-the-art com um Transformer pré-treinado (BERT);
- uma abordagem adicional utilizando Word2Vec.

## Dataset Escolhido

O dataset escolhido para esta análise é o FinancialPhraseBank, disponivel
no [Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news).

É um conjunto de dados relativamente popular utilizado na análise de sentimentos financeiros.
Este dataset contém frases extraídas de relatórios financeiros e principalmente de artigos de notícias.

É um dataset relativamente simples e pré-processado (até certo ponto, é necessário algumas transformações,
principalmente para nossos modelos). Não é tão balanceado (59% positivas, 28% neutras e 12% negativas).

# Abordagem Adotada

Para a abordagem baseline, utilizamos um modelo Bag of Words (BoW), que transforma cada frase em um vetor de frequências
de palavras, ignorando a ordem e o contexto das palavras, pois foi introduzido durante as aulas e também é usado
comumente para problemas beeem simples que envolvam poucas classes ou dados não muito complexos.

Para a abordagem state of the art (SOA) escolhemos uma arquitetura de Transformer que tem capacidade de processar
palavras em contexto, que é extremamente recorrente quando se trata de sentimentos em texto (o sentimento geralmente é
definido por algumas palavras mas que dependem extremamente do contexto, por exemplo: "O cenário econômico está
extremamente volátil, mas a Apple performou bem.", o sentimento é positivo ou negativo, depende se o contexto é o
mercado ou a empresa Apple).

Para a abordagem adicional, escolhemos o W2V, e utilizamos uma rede recorrente (LSTM, especificamente) para a
classificação. Essa escolha se dá pois é um bom ponto intermediário para os dois outros modelos, além de utilizar redes
recorrentes que foi outro tópico abordado em aula.

## Bag of Words

A nossa implementação tem dois pontos importantes:

- A classe de vetorização, que utiliza uma função de contagem de frequência como entrada do modelo de classificação.
- O modelo de classificação em si, que é apenas uma rede neural totalmente conectada (pesos lineares) com uma saída em
  porcentagem.

Essa abordagem é extremamente simples, baseada no teorema de Bayes, em que a frequência é utilizada como o Prior.

![BOW vector](https://uc-r.github.io/public/images/analytics/feature-engineering/bow-image.png)

## Word To Vector

O W2V é em dificuldade de implementação um pouco mais complexo do que o BOW, devido à necessidade de criar um espaço
latente de embeddings (representação vetorial das palavras, ou seja, transformar as palavras em números).

Nossa implementação conta com o método de vetorização, similar ao do BOW, mas utilizado a biblioteca Gensim que tem uma
implementação para a criação do espaço automaticamente utilizando as frases existentes.

![W2V embeddings](https://cdn.coveo.com/images/w_1200,h_700,c_scale/v1707326301/blogprod/WordEmbeddings_106321438d/WordEmbeddings_106321438d.png?_i=AA)

## Transformer com Transfer Learning

O Transformer é um modelo de rede neural que foi introduzido em 2017 e é extremamente eficaz para tarefas de NLP, pois
consegue entender o contexto de uma palavra observando tanto o que vem antes quanto o que vem depois dela (bidirecional).

Para a nossa implementação, utilizamos o modelo Bert-Small, que é uma versão mais leve do BERT, mas com a mesma
arquitetura. Congelamos os pesos do modelo e inserimos camadas adicionais para a classificação, que é uma forma de
transfer learning, ou seja, treinar um modelo em cima de outro modelo já treinado.

# Especificações

## Bag of Words (BOW)

Existem duas principais abordagens: Continuous Bag of Words (CBOW) e Skip-gram. CBOW prevê uma palavra com base no seu
contexto, enquanto Skip-gram faz o inverso, prevendo o contexto a partir de uma palavra. 
A nossa implementação utiliza o CBOW, que para o nosso problema é mais eficaz,
pois acreditamos que o contexto da palavra é mais importante do que a palavra em si.

Inserimos uma sequência de camadas totalmente conectadas (uma DNN padrão) após a vetorização dos dados como modelo de
classificação para a label de sentimento.

### Hiperparâmetros:

- input_dim=2000
- hidden_dim=6

### Argumentos de treino:

- max_epochs=50
- batch_size=64
- num_workers=8
- lr=1e-6

## Word To Vector (W2V)

Para a implementação do W2V, utilizamos a biblioteca Gensim, que tem uma implementação pronta para a criação de
embeddings.

Na nossa abordagem utilizamos uma camada LSTM (Long- Short-Term Memory), ou seja, montamos uma rede recorrente como
parte do modelo de classificação, além das camadas lineares que tem como entrada a saída do CBOW.

### Hiperparâmetros:

- embedding_dim=512
- hidden_dim=256
- num_layers=8

### Argumetnos de treino:

- max_epochs=20
- batch_size=64
- num_workers=8
- lr=1e-5

## Transformer (Transfer Learning baseado em pesos do Bert-Small)

O BERT (Bidirectional Encoder Representations from Transformers) é um modelo de Transformer desenvolvido pelo Google,
mas escolhemos uma versão com menos pesos, mas a mesma arquitetura. Ele é treinado para entender o contexto de uma
palavra observando tanto o que vem antes quanto o que vem depois dela (bidirecional). 

Isso permite uma interpretação
artifical de contexto, o que torna BERT extremamente eficaz para tarefas de NLP e realizar Transfer Learning em cima.
Congelamos os pesos do modelo bert-small e inserimos camadas adicionais que foram treinadas em cima dos nossos dados.

### Hiperparâmetros:

- hidden_layers=3 (apenas as camadas não congeladas)

## Argumentos de treino:

- max-epochs=10 (o modelo demora bastante para ser treinado)
- batch_size=64
- num_workers=8
- lr=1e-4