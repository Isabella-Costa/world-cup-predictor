# ⚽ World Cup Predictor

Sistema inteligente que estima as chances de cada seleção avançar em cada fase da Copa do Mundo
> Projeto desenvolvido para a disciplina de Inteligência Artificial — demonstra a aplicação de técnicas de IA em um problema real, com interface interativa, explicação do modelo e métricas de avaliação.

---

## 📋 Sumário

- [Visão geral](#-visão-geral)
- [Como funciona](#-como-funciona)
- [Arquitetura do sistema](#-arquitetura-do-sistema)
- [Inteligência artificial implementada](#-inteligência-artificial-implementada)
- [Dados utilizados](#-dados-utilizados)
- [Estrutura do repositório](#-estrutura-do-repositório)
- [Como rodar o projeto](#-como-rodar-o-projeto)
- [Tecnologias](#-tecnologias)
- [Equipe](#-equipe)

---

## Visão geral

O **World Cup Predictor** é um sistema que responde à pergunta: *"Qual a chance da seleção X chegar até tal fase da Copa?"*

Em vez de prever o campeão diretamente, o sistema:

1. Usa uma **rede neural** para estimar a probabilidade de vitória, empate ou derrota em **cada partida possível** entre duas seleções.
2. Usa essas probabilidades para **simular a Copa inteira milhares de vezes** 
3. Agrega o resultado das simulações para calcular, de forma estatística, a chance real de cada seleção alcançar oitavas, quartas, semifinal, final e título.

---

## Como funciona

**Fluxo do usuário:**

1. Seleciona as seleções participantes da Copa.
2. Escolhe uma seleção (ou uma etapa do torneio) para analisar.
3. Clica em **Calcular**.

**Saída do sistema:**

```
Seleção: Brasil
Grupos     ██████████ 92%
Oitavas    ██████████ 92%
Quartas    ████████   68%
Semifinal  █████      41%
Final      ███        23%
Campeão    ██         11%
```

Também é possível inverter a consulta: escolher uma **etapa** (ex: Final) e ver o ranking de todas as seleções com maior chance de chegar lá.

---

## Arquitetura do sistema

O sistema segue uma arquitetura de **sistema baseado em conhecimento**, dividida em três camadas:

- **Base de conhecimento**: dados históricos de Copas do Mundo (partidas, seleções, perfis) usados para treinar e alimentar o modelo.
- **Motor de inferência**: rede neural (nível partida) + simulação nível torneio.
- **Interface**: camada de apresentação em xxxx, onde o usuário interage com o sistema.

---

## Inteligência artificial implementada

### 1. Pré-processamento
- Junção dos dados de partidas históricas com o perfil de cada seleção na respectiva edição (ranking FIFA, valor de mercado do elenco, forma recente, histórico de fases alcançadas).
- Tratamento de valores nulos, normalização de variáveis numéricas e padronização de nomes de seleções entre as bases.

### 2. Rede Neural
- **Entradas:** atributos comparativos entre Time A e Time B (diferença de ranking FIFA, diferença de valor de elenco, forma recente, histórico de participações, etc.).
- **Saída:** probabilidade de `Vitória A`, `Empate` e `Vitória B`.
- Arquitetura, hiperparâmetros e justificativa de escolha do modelo documentados no relatório técnico.




### 3. Métricas de avaliação
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Matriz de confusão para o classificador de resultado de partida.**

---

## 📊 Dados utilizados

| Fonte | Conteúdo | Uso no projeto |
|---|---|---|
| `matches` | ~900 partidas reais de Copas do Mundo (1930–2018) | Treino do classificador de resultado de partida |
| `teams` | Seleções e suas confederações | Feature de confederação/região |
| `tournaments` | Metadados de cada edição (formato, sede, campeão) | Contexto estrutural por edição |
| Perfil por seleção/edição | Ranking FIFA, valor de elenco, forma recente, histórico (2002–2022) | Features de força de cada seleção |
| Perfil 2026 (sem rótulo) | Seleções participantes da Copa de 2026 | Base para a simulação final |

---

## 📁 Estrutura do repositório

```
world-cup-predictor/
├── data/
│   ├── raw/                # Dados originais (matches, teams, tournaments, perfis)
│   └── processed/          # Dados tratados e prontos para o modelo
├── notebooks/
│   └── eda.ipynb           # Análise exploratória dos dados
├── src/
│   ├── data_preparation.py    # Limpeza e junção dos datasets
│   ├── data_prep_jogador.py   # Preparação dos dados de jogadores
│   ├── data_validation.py   # Validação dos dados de jogos
├── app/
│   └── .py    # Interface do usuário
├── requirements.txt
└── README.md
```

---

## ▶️ Como rodar o projeto

```bash
# Clonar o repositório
git clone [https://github.com/<usuario>/world-cup-predictor.git]
cd world-cup-predictor

# Criar ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Treinar o modelo (
python src/train_model.py

# Rodar a interface
streamlit run app/streamlit_app.py
```

---

## 🛠️ Tecnologias

- **Python**
- **Pandas / NumPy** — manipulação de dados
- **scikit-learn** — métricas de avaliação

---

## 👥 Equipe


---

## 📄 Licença

Projeto acadêmico, desenvolvido para fins educacionais.
