# Pipeline de Dados - ETL para Classificação de Notícias da UFV

A capacidade de extrair valor dos dados se tornou fundamental em diversas áreas, especialmente em aplicações envolvendo Inteligência Artificial e aprendizado de máquina. Para que os modelos de Machine Learning consigam entregar resultados satisfatórios, é essencial que sejam alimentados por dados bem estruturados, confiáveis e relevantes. É nesse contexto que surge o conceito de **pipeline de dados**.

Um **pipeline de dados** pode ser entendido como um conjunto de etapas sequenciais automatizadas que garantem a coleta, organização, tratamento e utilização dos dados para um fim específico, como treinamento de modelos preditivos. Entre os diversos tipos de pipeline, destaca-se a metodologia ETL, acrônimo para Extract (Extrair), Transform (Transformar) e Load (Carregar).

## O que é ETL?

A abordagem ETL define três etapas claras e consecutivas que organizam o fluxo dos dados desde sua fonte original até o uso final em sistemas de análise ou aprendizado de máquina:

- **Extração (Extract)**: corresponde à obtenção inicial dos dados, vindos de fontes diversas como sites, APIs, bancos de dados ou arquivos. Essa etapa é especialmente crítica, pois a qualidade dos dados obtidos define a eficácia do pipeline.

- **Transformação (Transform)**: envolve todas as operações realizadas sobre os dados extraídos com o objetivo de organizá-los, validá-los e convertê-los para formatos apropriados ao uso desejado. Processos comuns são limpeza, normalização e criação de representações numéricas (como vetores de embeddings).

- **Carregamento (Load)**: é o processo pelo qual os dados já transformados são inseridos ou "carregados" em um sistema alvo, como um modelo de aprendizado de máquina ou um banco de dados preparado para consultas rápidas e eficientes.

Neste módulo, aprofundaremos a etapa inicial do processo ETL, discutindo técnicas práticas e ferramentas de extração de dados. Nosso exemplo prático será um sistema para classificação automática de notícias obtidas diretamente do site da Universidade Federal de Viçosa (UFV).

## Extract: Técnicas e Práticas de Extração de Dados

Antes que um modelo de Machine Learning possa realizar qualquer tipo de tarefa, é necessário coletar informações relevantes. A extração eficiente de dados é uma tarefa complexa que demanda técnicas específicas, ferramentas adequadas e conhecimento das limitações legais e éticas envolvidas.

### O que é Web Crawling e Web Scraping?

**Web Crawling** refere-se ao processo sistemático de navegar por páginas da web automaticamente, descobrindo links e páginas que possam conter informações de interesse. Uma vez identificadas essas páginas, é comum aplicar técnicas de **Web Scraping** para extração dos conteúdos específicos.

**Web Scraping** é o ato de coletar dados diretamente do HTML ou conteúdo visual das páginas da web. Trata-se de uma prática amplamente utilizada, porém sensível a mudanças no layout dos sites, restrições de acesso e implicações legais.

#### Questões Éticas e Legais do Web Scraping

Embora seja tecnicamente simples, é essencial considerar aspectos éticos e legais do Web Scraping:

- **robots.txt**: arquivo disponibilizado pelos sites indicando quais partes podem ou não ser acessadas por crawlers.
- **Copyright e propriedade intelectual**: os dados extraídos podem estar protegidos por direitos autorais, sendo necessário obter permissão explícita ou utilizar apenas informações públicas.

### Ferramentas de Extração

As ferramentas para Web Scraping são divididas principalmente em duas categorias:

#### Navegadores Automatizados (Headless Browsers)

Ferramentas como **Selenium** e **Playwright** são utilizadas para simular interações humanas com navegadores sem interface gráfica (headless). São ideais para lidar com páginas dinâmicas, carregadas por JavaScript.

#### Requisições HTTP e Parsing de HTML

Bibliotecas como **httpx** permitem fazer requisições HTTP diretamente aos servidores, obtendo respostas rapidamente sem precisar carregar páginas completas. Para a análise e extração de informações específicas do HTML, ferramentas como **BeautifulSoup** são amplamente usadas pela simplicidade e robustez.

### Validação e Armazenamento dos Dados

Após extrair dados, é importante validá-los. Ferramentas como **Pydantic** garantem que os dados estejam corretos e consistentes antes de serem armazenados.

Para armazenamento, geralmente se utiliza um banco de dados relacional. A utilização de um ORM (Object-Relational Mapping) como o **peewee** facilita essa tarefa ao permitir trabalhar com os dados utilizando objetos e métodos da linguagem de programação, sem a necessidade de manipular diretamente o SQL.

### Exemplo Aplicado: Notícias do site da UFV

No projeto prático, criamos um scraper utilizando técnicas modernas como requisições assíncronas (httpx) para buscar notícias rapidamente e de forma simultânea. O exemplo de código do projeto demonstra como implementar eficientemente essa abordagem.

> Veja o código completo do scraper no repositório do projeto para detalhes específicos da implementação assíncrona e da utilização do BeautifulSoup.

### Armazenando Dados com ORM

O ORM facilita operações comuns como inserir, consultar e atualizar dados, abstraindo as consultas SQL diretas e tornando o desenvolvimento mais produtivo e intuitivo. No projeto, utilizamos o banco de dados SQLite, adequado para ambientes pequenos e desenvolvimento rápido.

> Para detalhes específicos sobre a implementação com peewee, consulte o arquivo `models.py` no repositório do projeto.

---

## Transform
[TODO]

## Load
[TODO]

---
