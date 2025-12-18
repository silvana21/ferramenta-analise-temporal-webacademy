# Ferramenta de Análise Temporal de Regras de Associação

Este repositório contém uma ferramenta desenvolvida no contexto de um **projeto de doutorado em Engenharia de Software**, cujo objetivo é **extrair e analisar regras de associação em dados de pull requests**, observando a **variação temporal das medidas de interesse** (como suporte, confiança e lift).

A ferramenta foi desenvolvida em **Python**, com **interface web** via Streamlit, e permite que pesquisadores e estudantes realizem análises exploratórias e comparativas a partir de arquivos CSV.

---

## Objetivo da Ferramenta

* Extrair regras de associação a partir de uma base completa de dados.
* Particionar os dados temporalmente (janelas fixas ou marcos definidos pelo usuário).
* Comparar medidas das regras entre a base geral e as partições.
* Identificar variações relevantes ao longo do tempo.
* Apoiar análises qualitativas posteriores em projetos de software.

---

## Pré-requisitos

Para executar a ferramenta localmente, é necessário ter:

### 1. Software Básico

* **Python 3.9 ou superior**
* **Git** (para clonar o repositório)
* Sistema operacional:

  * Windows 10 ou superior

---

## Bibliotecas Python Utilizadas

As principais bibliotecas usadas no projeto são:

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `mlxtend` (Apriori e regras de associação)
* `streamlit` (interface web)

---

## Configuração do Ambiente

### 1. Clonar o repositório

```bash
git clone https://github.com/silvana21/ferramenta-analise-temporal-webacademy.git
cd ferramenta-analise-temporal-webacademy
```

### 2. (Opcional, mas recomendado) Criar ambiente virtual

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```


### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

---

## Executando a Ferramenta

A aplicação utiliza **Streamlit**.

Para iniciar o servidor local:

```bash
streamlit run main.py
```

Após isso, o navegador abrirá automaticamente (ou acesse manualmente):

```
http://localhost:8501
```

---

## Entrada de Dados

A ferramenta espera um arquivo **CSV** contendo informações sobre pull requests. Mas não limitado à este tipo de dado.

Exemplos de atributos normalmente utilizados:

* Data de criação do pull request
* Autor
* Status (aceito/rejeitado)
* Tempo de vida
* Tipo de contribuição
* Indicadores relacionados a contribuição externa

---

## Funcionalidades Disponíveis

* Upload de arquivo CSV
* Configuração de parâmetros do algoritmo Apriori
* Extração de regras da base geral
* Particionamento temporal dos dados
* Extração das mesmas regras nas partições
* Comparação de suporte, confiança e lift
* Visualização gráfica das variações

---

## Contexto Educacional

Este repositório será utilizado por **alunos**, que irão:

* Estudar conceitos de mineração de dados e regras de associação
* Entender análise temporal aplicada a dados de engenharia de software
* Reconstruir/extender a ferramenta (ex: versão em Java, autenticação, histórico por usuário)

O código está organizado de forma **didática**, visando facilitar a compreensão e evolução do projeto.

---

## Conceitos Relacionados

* Mineração de Dados
* Regras de Associação
* Algoritmo Apriori
* Análise Temporal (baseada em partições)
* Engenharia de Software Empírica

---

## 📄 Licença

Este projeto é disponibilizado apenas para **fins acadêmicos e educacionais**.

---

## Contato

Em caso de dúvidas ou sugestões, utilize as *Issues* do GitHub ou entre em contato com a autora do projeto.

---

Bons estudos e boas análises!
