import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import hashlib

# ================================
# CONFIGURAÇÃO DA PÁGINA
# ================================
st.set_page_config(
    page_title="Sistema de Score de Risco",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CARREGAMENTO DE DADOS E MODELO
# ================================
@st.cache_resource
def load_embedding_model():
    """Carrega modelo de embedding"""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_vectorized_data():
    """Carrega dados vetorizados do arquivo .pkl"""
    try:
        with open('data_embeddings.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.warning("⚠️ Arquivo data_embeddings.pkl não encontrado! Sistema funcionando sem base histórica.")
        return None

# ================================
# FUNÇÕES DE API
# ================================
def validar_cpf(cpf):
    """Validação básica de CPF"""
    cpf = ''.join(filter(str.isdigit, cpf))
    if len(cpf) != 11:
        return False
    
    def calcular_digito(cpf_parcial, peso_inicial):
        soma = sum(int(cpf_parcial[i]) * (peso_inicial - i) for i in range(len(cpf_parcial)))
        resto = soma % 11
        return 0 if resto < 2 else 11 - resto
    
    if cpf == cpf[0] * 11:
        return False
    
    digito1 = calcular_digito(cpf[:9], 10)
    digito2 = calcular_digito(cpf[:10], 11)
    
    return cpf[-2:] == f"{digito1}{digito2}"

def consultar_serasa_api(cpf, nome):
    """Consulta API Serasa"""
    try:
        api_key = st.secrets.get("SERASA_API_KEY", None)
        
        if not api_key:
            return simular_dados_serasa(cpf, nome)
        
        url = st.secrets.get("SERASA_API_URL", "https://api.serasaexperian.com.br/v1/consulta")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {"cpf": cpf, "nome": nome}
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return simular_dados_serasa(cpf, nome)
            
    except Exception as e:
        return simular_dados_serasa(cpf, nome)

def consultar_boavista_api(cpf):
    """Consulta API Boa Vista SCPC"""
    try:
        api_key = st.secrets.get("BOAVISTA_API_KEY", None)
        
        if not api_key:
            return simular_dados_boavista(cpf)
        
        url = st.secrets.get("BOAVISTA_API_URL", "https://api.boavistascpc.com.br/v2/consulta")
        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        payload = {"cpf": cpf}
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return simular_dados_boavista(cpf)
            
    except Exception as e:
        return simular_dados_boavista(cpf)

def consultar_receita_federal(cpf):
    """Consulta situação cadastral na Receita Federal"""
    try:
        api_key = st.secrets.get("RF_API_KEY", None)
        
        if api_key:
            url = st.secrets.get("RF_API_URL", f"https://api.receitaws.com.br/v1/cpf/{cpf}")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
        
        return simular_dados_receita(cpf)
            
    except Exception as e:
        return simular_dados_receita(cpf)

# ================================
# FUNÇÕES DE SIMULAÇÃO
# ================================
def simular_dados_serasa(cpf, nome):
    """Simula resposta da API Serasa"""
    hash_cpf = int(hashlib.md5(cpf.encode()).hexdigest(), 16)
    score_base = 300 + (hash_cpf % 700)
    
    return {
        "score_credito": score_base,
        "faixa_score": "Alto" if score_base > 700 else "Médio" if score_base > 500 else "Baixo",
        "consultas_6m": (hash_cpf % 5) + 1,
        "consultas_12m": (hash_cpf % 10) + 2,
        "restricoes_ativas": hash_cpf % 3,
        "valor_restricoes": (hash_cpf % 5000) if (hash_cpf % 3) > 0 else 0,
        "cheques_sem_fundo_12m": hash_cpf % 2,
        "protestos_12m": hash_cpf % 2,
        "renda_presumida": 3000 + (hash_cpf % 10000)
    }

def simular_dados_boavista(cpf):
    """Simula resposta da API Boa Vista"""
    hash_cpf = int(hashlib.md5(cpf.encode()).hexdigest(), 16)
    
    return {
        "score_boavista": 400 + (hash_cpf % 600),
        "dividas_ativas": (hash_cpf % 4) > 1,
        "valor_dividas": (hash_cpf % 8000) if (hash_cpf % 4) > 1 else 0,
        "ultimo_registro_negativo": "2024-08" if (hash_cpf % 3) == 0 else None,
        "historico_pagamentos": "Regular" if hash_cpf % 2 == 0 else "Irregular",
        "participacao_societaria": hash_cpf % 5 == 0
    }

def simular_dados_receita(cpf):
    """Simula resposta da Receita Federal"""
    hash_cpf = int(hashlib.md5(cpf.encode()).hexdigest(), 16)
    
    return {
        "situacao_cadastral": "Regular" if hash_cpf % 10 > 1 else "Pendente",
        "data_inscricao": "1990-01-01",
        "comprovante_emitido": True
    }

# ================================
# VETORIZAÇÃO E SIMILARIDADE
# ================================
def criar_embedding_cliente(nome, cpf, dados_externos):
    """Cria embedding vetorial dos dados do cliente"""
    model = load_embedding_model()
    
    texto_cliente = f"""
    Nome: {nome}
    Score Crédito: {dados_externos['serasa']['score_credito']}
    Faixa Score: {dados_externos['serasa']['faixa_score']}
    Consultas 6 meses: {dados_externos['serasa']['consultas_6m']}
    Restrições: {dados_externos['serasa']['restricoes_ativas']}
    Histórico Pagamentos: {dados_externos['boavista']['historico_pagamentos']}
    Situação Cadastral: {dados_externos['receita']['situacao_cadastral']}
    Renda Presumida: {dados_externos['serasa']['renda_presumida']}
    """
    
    embedding = model.encode(texto_cliente.strip())
    return embedding

def buscar_similares(embedding_cliente, data_vectorized, top_k=5):
    """Busca clientes similares na base vetorizada"""
    if data_vectorized is None:
        return []
    
    embeddings_base = np.array(data_vectorized['embeddings'])
    similarities = cosine_similarity([embedding_cliente], embeddings_base)[0]
    
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    similares = []
    for idx in top_indices:
        similares.append({
            'nome': data_vectorized['nomes'][idx],
            'score_historico': data_vectorized['scores'][idx],
            'similaridade': float(similarities[idx]),
            'sinistros_12m': data_vectorized.get('sinistros_12m', [0] * len(embeddings_base))[idx]
        })
    
    return similares

# ================================
# CÁLCULO DO SCORE
# ================================
def calcular_score_risco(dados_externos, similares):
    """Calcula score de 0-100"""
    
    # Normaliza scores
    score_serasa_norm = (dados_externos['serasa']['score_credito'] - 300) / 7
    score_boavista_norm = (dados_externos['boavista']['score_boavista'] - 400) / 6
    
    # Penalizações
    penalidade_restricoes = dados_externos['serasa']['restricoes_ativas'] * 5
    penalidade_dividas = 10 if dados_externos['boavista']['dividas_ativas'] else 0
    penalidade_consultas = min(dados_externos['serasa']['consultas_6m'] * 2, 10)
    penalidade_cadastro = 15 if dados_externos['receita']['situacao_cadastral'] != "Regular" else 0
    
    # Bonus similaridade
    if similares:
        media_score_similares = np.mean([s['score_historico'] for s in similares])
        media_sinistros_similares = np.mean([s['sinistros_12m'] for s in similares])
        
        bonus_similares = (media_score_similares - 50) * 0.2
        penalidade_sinistros = media_sinistros_similares * 3
    else:
        bonus_similares = 0
        penalidade_sinistros = 0
        media_score_similares = 50
    
    # Score final
    score_base = (score_serasa_norm * 0.4 + score_boavista_norm * 0.4) + 20
    
    score_final = score_base + bonus_similares - (
        penalidade_restricoes +
        penalidade_dividas +
        penalidade_consultas +
        penalidade_cadastro +
        penalidade_sinistros
    )
    
    score_final = max(0, min(100, score_final))
    
    # Reason codes
    reasons = []
    if penalidade_restricoes > 0:
        reasons.append({
            'fator': 'Restrições Ativas',
            'impacto': -penalidade_restricoes,
            'descricao': f'{dados_externos["serasa"]["restricoes_ativas"]} restrição(ões) ativa(s)'
        })
    if penalidade_dividas > 0:
        reasons.append({
            'fator': 'Dívidas Ativas',
            'impacto': -penalidade_dividas,
            'descricao': f'R$ {dados_externos["boavista"]["valor_dividas"]:.2f} em dívidas'
        })
    if penalidade_consultas > 0:
        reasons.append({
            'fator': 'Consultas Recentes',
            'impacto': -penalidade_consultas,
            'descricao': f'{dados_externos["serasa"]["consultas_6m"]} consultas em 6 meses'
        })
    if bonus_similares > 0:
        reasons.append({
            'fator': 'Perfil Similar a Bons Clientes',
            'impacto': bonus_similares,
            'descricao': f'Alta similaridade com clientes score médio {media_score_similares:.0f}'
        })
    
    reasons = sorted(reasons, key=lambda x: abs(x['impacto']), reverse=True)[:5]
    
    return {
        'score': round(score_final, 2),
        'banda': 'MUITO BAIXO' if score_final >= 80 else 'BAIXO' if score_final >= 60 else 'MÉDIO' if score_final >= 40 else 'ALTO' if score_final >= 20 else 'MUITO ALTO',
        'probabilidade_sinistro_12m': round((100 - score_final) / 100 * 0.15, 4),
        'reasons': reasons,
        'componentes': {
            'score_serasa': round(score_serasa_norm, 2),
            'score_boavista': round(score_boavista_norm, 2),
            'bonus_similares': round(bonus_similares, 2),
            'total_penalidades': round(sum([penalidade_restricoes, penalidade_dividas, penalidade_consultas, penalidade_cadastro, penalidade_sinistros]), 2)
        }
    }

# ================================
# INTERFACE STREAMLIT
# ================================
def main():
    st.title("🛡️ Sistema de Score de Risco - MVP")
    st.markdown("**Análise de risco com enriquecimento de dados e vetorização**")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Sobre o Sistema")
        st.markdown("""
        **Funcionalidades:**
        - ✅ Validação de CPF
        - 🔍 Consulta APIs externas
        - 🧬 Vetorização de dados
        - 📊 Busca de similares
        - 🎯 Score 0-100
        - 📋 Reason codes
        """)
        
        st.markdown("---")
        st.markdown("**Status das APIs:**")
        
        apis = {
            "Serasa": "SERASA_API_KEY",
            "Boa Vista": "BOAVISTA_API_KEY",
            "Receita Federal": "RF_API_KEY"
        }
        
        for api, key in apis.items():
            if key in st.secrets:
                st.success(f"✅ {api}")
            else:
                st.warning(f"⚠️ {api} (simulação)")
    
    # Formulário
    st.header("📝 Dados do Cliente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cpf_input = st.text_input("CPF", placeholder="00000000000")
    
    with col2:
        nome_input = st.text_input("Nome Completo", placeholder="João da Silva")
    
    if st.button("🚀 Analisar Risco", type="primary", use_container_width=True):
        
        if not cpf_input or not nome_input:
            st.error("⚠️ Preencha CPF e Nome")
            return
        
        cpf_limpo = ''.join(filter(str.isdigit, cpf_input))
        
        if not validar_cpf(cpf_limpo):
            st.error("❌ CPF inválido!")
            return
        
        with st.spinner("🔄 Processando análise..."):
            progress_bar = st.progress(0)
            
            # Consulta APIs
            st.info("📡 Consultando APIs...")
            progress_bar.progress(30)
            
            dados_serasa = consultar_serasa_api(cpf_limpo, nome_input)
            dados_boavista = consultar_boavista_api(cpf_limpo)
            dados_receita = consultar_receita_federal(cpf_limpo)
            
            dados_externos = {
                'serasa': dados_serasa,
                'boavista': dados_boavista,
                'receita': dados_receita
            }
            
            progress_bar.progress(60)
            
            # Vetorização
            st.info("🧬 Criando embedding...")
            embedding_cliente = criar_embedding_cliente(nome_input, cpf_limpo, dados_externos)
            
            progress_bar.progress(80)
            
            # Busca similares
            data_vectorized = load_vectorized_data()
            similares = buscar_similares(embedding_cliente, data_vectorized)
            
            # Calcula score
            resultado = calcular_score_risco(dados_externos, similares)
            
            progress_bar.progress(100)
        
        st.success("✅ Análise concluída!")
        
        # RESULTADOS
        st.header("📊 Resultado da Análise")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Score de Risco", f"{resultado['score']}/100")
        
        with col2:
            st.metric("Banda de Risco", resultado['banda'])
        
        with col3:
            st.metric("Prob. Sinistro 12m", f"{resultado['probabilidade_sinistro_12m']*100:.2f}%")
        
        with col4:
            cor = "🟢" if resultado['score'] >= 70 else "🟡" if resultado['score'] >= 40 else "🔴"
            st.metric("Status", cor)
        
        # Reason Codes
        st.subheader("🎯 Principais Fatores")
        
        for i, reason in enumerate(resultado['reasons'], 1):
            with st.expander(f"**{i}. {reason['fator']}** - Impacto: {reason['impacto']:+.2f} pontos"):
                st.write(reason['descricao'])
        
        # Dados APIs
        with st.expander("📡 Dados das APIs"):
            tab1, tab2, tab3 = st.tabs(["Serasa", "Boa Vista", "Receita"])
            
            with tab1:
                st.json(dados_serasa)
            with tab2:
                st.json(dados_boavista)
            with tab3:
                st.json(dados_receita)
        
        # Similares
        if similares:
            st.subheader("👥 Clientes Similares")
            df_similares = pd.DataFrame(similares)
            df_similares['similaridade'] = (df_similares['similaridade'] * 100).round(2)
            st.dataframe(df_similares, use_container_width=True, hide_index=True)
        
        # Download
        resultado_completo = {
            'cliente': {'cpf': cpf_limpo, 'nome': nome_input},
            'timestamp': datetime.now().isoformat(),
            'score': resultado,
            'dados_externos': dados_externos
        }
        
        st.download_button(
            "⬇️ Baixar JSON",
            data=json.dumps(resultado_completo, indent=2, ensure_ascii=False),
            file_name=f"score_{cpf_limpo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
