import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import requests
import json
import re
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
# FUNÇÕES BRASILAPI
# ================================
BASE_URL_BRASILAPI = "https://brasilapi.com.br/api"
TIMEOUT_API = 10

def normalizar_cnpj(cnpj: str) -> str:
    """Remove caracteres não numéricos do CNPJ"""
    cnpj_limpo = re.sub(r'\D', '', cnpj)
    if len(cnpj_limpo) != 14:
        raise ValueError("CNPJ deve ter 14 dígitos")
    return cnpj_limpo

def normalizar_cep(cep: str) -> str:
    """Remove caracteres não numéricos do CEP"""
    cep_limpo = re.sub(r'\D', '', cep)
    if len(cep_limpo) != 8:
        raise ValueError("CEP deve ter 8 dígitos")
    return cep_limpo

def parse_valor_brl(valor_str: str) -> float:
    """Converte 'R$ 45.000,00' para float"""
    if not valor_str:
        return 0.0
    valor_limpo = valor_str.replace('R$', '').replace(' ', '').replace('.', '').replace(',', '.')
    try:
        return float(valor_limpo)
    except:
        return 0.0

def consultar_cnpj_brasilapi(cnpj: str):
    """Consulta dados cadastrais de CNPJ"""
    try:
        cnpj_limpo = normalizar_cnpj(cnpj)
        url = f"{BASE_URL_BRASILAPI}/cnpj/v1/{cnpj_limpo}"
        response = requests.get(url, timeout=TIMEOUT_API)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'cnpj': data.get('cnpj'),
                'razao_social': data.get('razao_social'),
                'nome_fantasia': data.get('nome_fantasia'),
                'situacao_cadastral': data.get('descricao_situacao_cadastral'),
                'data_inicio_atividade': data.get('data_inicio_atividade'),
                'cnae_principal': data.get('cnae_fiscal'),
                'cnae_descricao': data.get('cnae_fiscal_descricao'),
                'porte': data.get('porte'),
                'cep': data.get('cep'),
                'uf': data.get('uf'),
                'municipio': data.get('municipio'),
                'bairro': data.get('bairro'),
                'logradouro': data.get('logradouro'),
                'qsa': data.get('qsa', []),
                'status': 'success'
            }
        return {'status': 'not_found'}
    except:
        return {'status': 'error'}

def consultar_cep_brasilapi(cep: str):
    """Consulta CEP com geolocalização"""
    try:
        cep_limpo = normalizar_cep(cep)
        
        # Tenta v2 primeiro (com coordenadas)
        url = f"{BASE_URL_BRASILAPI}/cep/v2/{cep_limpo}"
        response = requests.get(url, timeout=TIMEOUT_API)
        
        if response.status_code == 200:
            data = response.json()
            location = data.get('location', {})
            coords = location.get('coordinates', [None, None]) if location else [None, None]
            
            return {
                'cep': data.get('cep'),
                'uf': data.get('state'),
                'municipio': data.get('city'),
                'bairro': data.get('neighborhood'),
                'logradouro': data.get('street'),
                'longitude': coords[0],
                'latitude': coords[1],
                'geo_disponivel': coords[0] is not None,
                'status': 'success'
            }
        
        # Fallback v1 (sem coordenadas)
        url = f"{BASE_URL_BRASILAPI}/cep/v1/{cep_limpo}"
        response = requests.get(url, timeout=TIMEOUT_API)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'cep': data.get('cep'),
                'uf': data.get('state'),
                'municipio': data.get('city'),
                'bairro': data.get('neighborhood'),
                'logradouro': data.get('street'),
                'longitude': None,
                'latitude': None,
                'geo_disponivel': False,
                'status': 'success'
            }
        
        return {'status': 'not_found'}
    except:
        return {'status': 'error'}

def consultar_fipe_brasilapi(marca: str, modelo: str):
    """Busca valor FIPE de um veículo"""
    try:
        # Busca tabela atual
        url_tabelas = f"{BASE_URL_BRASILAPI}/fipe/tabelas/v1"
        resp_tab = requests.get(url_tabelas, timeout=TIMEOUT_API)
        if resp_tab.status_code != 200:
            return {'status': 'error'}
        
        tabelas = resp_tab.json()
        tabela_ref = str(tabelas[-1]['codigo'])
        
        # Busca marcas
        url_marcas = f"{BASE_URL_BRASILAPI}/fipe/marcas/v1/carros"
        resp_marcas = requests.get(url_marcas, params={'tabela_referencia': tabela_ref}, timeout=TIMEOUT_API)
        if resp_marcas.status_code != 200:
            return {'status': 'error'}
        
        marcas = resp_marcas.json()
        codigo_marca = None
        
        for m in marcas:
            if marca.lower() in m['nome'].lower():
                codigo_marca = m['valor']
                break
        
        if not codigo_marca:
            return {'status': 'not_found', 'message': 'Marca não encontrada'}
        
        # Busca modelos
        url_modelos = f"{BASE_URL_BRASILAPI}/fipe/veiculos/v1/carros/{codigo_marca}"
        resp_mod = requests.get(url_modelos, params={'tabela_referencia': tabela_ref}, timeout=TIMEOUT_API)
        if resp_mod.status_code != 200:
            return {'status': 'error'}
        
        modelos = resp_mod.json()
        codigo_fipe = None
        
        for mod in modelos:
            if modelo.lower() in mod['nome'].lower():
                codigo_fipe = mod.get('codigo')
                break
        
        if not codigo_fipe:
            return {'status': 'not_found', 'message': 'Modelo não encontrado'}
        
        # Busca preço
        url_preco = f"{BASE_URL_BRASILAPI}/fipe/preco/v1/{codigo_fipe}"
        resp_preco = requests.get(url_preco, params={'tabela_referencia': tabela_ref}, timeout=TIMEOUT_API)
        if resp_preco.status_code != 200:
            return {'status': 'error'}
        
        data = resp_preco.json()[0]
        valor_str = data.get('valor', 'R$ 0,00')
        
        return {
            'valor_formatado': valor_str,
            'valor_numerico': parse_valor_brl(valor_str),
            'marca': data.get('marca'),
            'modelo': data.get('modelo'),
            'ano_modelo': data.get('anoModelo'),
            'combustivel': data.get('combustivel'),
            'mes_referencia': data.get('mesReferencia'),
            'status': 'success'
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def calcular_idade_empresa(data_inicio: str):
    """Calcula idade da empresa em anos"""
    try:
        inicio = datetime.strptime(data_inicio, '%Y-%m-%d')
        hoje = datetime.now()
        delta = hoje - inicio
        return round(delta.days / 365.25, 2)
    except:
        return None

def calcular_ajuste_cnpj(dados_cnpj):
    """Calcula ajustes no score baseado em CNPJ"""
    if dados_cnpj.get('status') != 'success':
        return {'ajuste': 0, 'reasons': []}
    
    ajuste = 0
    reasons = []
    
    # Situação cadastral
    situacao = dados_cnpj.get('situacao_cadastral', '')
    if situacao == 'ATIVA':
        idade = calcular_idade_empresa(dados_cnpj.get('data_inicio_atividade', ''))
        if idade and idade >= 10:
            ajuste += 5
            reasons.append(f"Empresa ativa há {idade:.1f} anos (+5 pts)")
        elif idade and idade >= 5:
            ajuste += 3
            reasons.append(f"Empresa ativa há {idade:.1f} anos (+3 pts)")
    else:
        ajuste -= 10
        reasons.append(f"Empresa em situação: {situacao} (-10 pts)")
    
    # Porte
    if dados_cnpj.get('porte') == 'DEMAIS':
        ajuste += 2
        reasons.append("Empresa de grande porte (+2 pts)")
    
    return {'ajuste': ajuste, 'reasons': reasons}

def calcular_ajuste_fipe(dados_fipe):
    """Calcula ajustes no score baseado no valor FIPE"""
    if dados_fipe.get('status') != 'success':
        return {'ajuste': 0, 'reasons': []}
    
    ajuste = 0
    reasons = []
    valor = dados_fipe.get('valor_numerico', 0)
    
    # Severidade por valor
    if valor >= 100000:
        ajuste -= 8
        reasons.append(f"Veículo alto valor FIPE: R$ {valor:,.2f} (-8 pts)")
    elif valor >= 60000:
        ajuste -= 5
        reasons.append(f"Veículo valor elevado FIPE: R$ {valor:,.2f} (-5 pts)")
    elif valor >= 30000:
        ajuste -= 2
        reasons.append(f"Veículo valor médio FIPE: R$ {valor:,.2f} (-2 pts)")
    else:
        reasons.append(f"Veículo valor FIPE: R$ {valor:,.2f}")
    
    return {'ajuste': ajuste, 'reasons': reasons}

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
# FUNÇÕES DE API TRADICIONAIS
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
    """Consulta API Serasa (com fallback simulado)"""
    try:
        api_key = st.secrets.get("SERASA_API_KEY", None)
        
        if not api_key:
            return simular_dados_serasa(cpf, nome)
        
        url = st.secrets.get("SERASA_API_URL", "")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"cpf": cpf, "nome": nome}
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return simular_dados_serasa(cpf, nome)
    except:
        return simular_dados_serasa(cpf, nome)

def consultar_boavista_api(cpf):
    """Consulta API Boa Vista (com fallback simulado)"""
    try:
        api_key = st.secrets.get("BOAVISTA_API_KEY", None)
        
        if not api_key:
            return simular_dados_boavista(cpf)
        
        url = st.secrets.get("BOAVISTA_API_URL", "")
        headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        payload = {"cpf": cpf}
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return simular_dados_boavista(cpf)
    except:
        return simular_dados_boavista(cpf)

def consultar_receita_federal(cpf):
    """Consulta situação cadastral na Receita Federal (com fallback simulado)"""
    try:
        api_key = st.secrets.get("RF_API_KEY", None)
        
        if api_key:
            url = st.secrets.get("RF_API_URL", f"https://api.receitaws.com.br/v1/cpf/{cpf}")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
        
        return simular_dados_receita(cpf)
    except:
        return simular_dados_receita(cpf)

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
        'reasons': reasons
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
        - 🔍 Consulta APIs de crédito
        - 🌐 Enriquecimento BrasilAPI
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
        
        st.success("✅ BrasilAPI (pública)")
    
    # Formulário
    st.header("📝 Dados do Cliente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cpf_input = st.text_input("CPF *", placeholder="00000000000")
    
    with col2:
        nome_input = st.text_input("Nome Completo *", placeholder="João da Silva")
    
    # Campos opcionais BrasilAPI
    with st.expander("🌐 Enriquecimento BrasilAPI (Opcional)", expanded=False):
        st.markdown("*Adicione dados extras para melhorar a análise*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cnpj_input = st.text_input(
                "CNPJ Empregador/Empresa",
                placeholder="00000000000000",
                help="Para clientes PJ ou vincular empregador PF"
            )
            
            cep_input = st.text_input(
                "CEP Residência",
                placeholder="00000000",
                help="Para análise de risco geográfico"
            )
        
        with col2:
            usar_fipe = st.checkbox("📊 Consultar Valor FIPE", value=False)
            
            if usar_fipe:
                fipe_marca = st.text_input("Marca", placeholder="Ex: Volkswagen")
                fipe_modelo = st.text_input("Modelo", placeholder="Ex: Gol")
    
    # Botão análise
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
            
            # APIs tradicionais
            st.info("📡 Consultando APIs de crédito...")
            progress_bar.progress(30)
            
            dados_serasa = consultar_serasa_api(cpf_limpo, nome_input)
            dados_boavista = consultar_boavista_api(cpf_limpo)
            dados_receita = consultar_receita_federal(cpf_limpo)
            
            dados_externos = {
                'serasa': dados_serasa,
                'boavista': dados_boavista,
                'receita': dados_receita
            }
            
            progress_bar.progress(50)
            
            # BrasilAPI
            dados_brasilapi = {}
            ajustes_brasilapi = {'ajuste_total': 0, 'reasons': []}
            
            st.info("🌐 Consultando BrasilAPI...")
            
            # CNPJ
            if cnpj_input:
                try:
                    dados_cnpj = consultar_cnpj_brasilapi(cnpj_input)
                    if dados_cnpj.get('status') == 'success':
                        dados_brasilapi['cnpj'] = dados_cnpj
                        ajuste = calcular_ajuste_cnpj(dados_cnpj)
                        ajustes_brasilapi['ajuste_total'] += ajuste['ajuste']
                        ajustes_brasilapi['reasons'].extend(ajuste['reasons'])
                except:
                    pass
            
            # CEP
            if cep_input:
                try:
                    dados_cep = consultar_cep_brasilapi(cep_input)
                    if dados_cep.get('status') == 'success':
                        dados_brasilapi['cep'] = dados_cep
                except:
                    pass
            
            # FIPE
            if usar_fipe and fipe_marca and fipe_modelo:
                try:
                    dados_fipe = consultar_fipe_brasilapi(fipe_marca, fipe_modelo)
                    if dados_fipe.get('status') == 'success':
                        dados_brasilapi['fipe'] = dados_fipe
                        ajuste = calcular_ajuste_fipe(dados_fipe)
                        ajustes_brasilapi['ajuste_total'] += ajuste['ajuste']
                        ajustes_brasilapi['reasons'].extend(ajuste['reasons'])
                except:
                    pass
            
            progress_bar.progress(70)
            
            # Vetorização
            st.info("🧬 Criando embedding...")
            embedding_cliente = criar_embedding_cliente(nome_input, cpf_limpo, dados_externos)
            
            progress_bar.progress(80)
            
            # Similares
            data_vectorized = load_vectorized_data()
            similares = buscar_similares(embedding_cliente, data_vectorized)
            
            # Score
            resultado = calcular_score_risco(dados_externos, similares)
            
            # Aplica ajustes BrasilAPI
            if ajustes_brasilapi['ajuste_total'] != 0:
                resultado['score'] = max(0, min(100, resultado['score'] + ajustes_brasilapi['ajuste_total']))
                resultado['score'] = round(resultado['score'], 2)
                resultado['probabilidade_sinistro_12m'] = round((100 - resultado['score']) / 100 * 0.15, 4)
                
                # Reclassifica banda
                score_ajustado = resultado['score']
                resultado['banda'] = 'MUITO BAIXO' if score_ajustado >= 80 else 'BAIXO' if score_ajustado >= 60 else 'MÉDIO' if score_ajustado >= 40 else 'ALTO' if score_ajustado >= 20 else 'MUITO ALTO'
                
                # Adiciona reasons BrasilAPI
                for reason_text in ajustes_brasilapi['reasons']:
                    resultado['reasons'].append({
                        'fator': 'BrasilAPI',
                        'impacto': ajustes_brasilapi['ajuste_total'],
                        'descricao': reason_text
                    })
            
            progress_bar.progress(100)
        
        st.success("✅ Análise concluída!")
        
        # ================================
        # RESULTADOS
        # ================================
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
        
        # Dados APIs tradicionais
        with st.expander("📡 Dados das APIs de Crédito"):
            tab1, tab2, tab3 = st.tabs(["Serasa", "Boa Vista", "Receita Federal"])
            
            with tab1:
                st.json(dados_serasa)
            with tab2:
                st.json(dados_boavista)
            with tab3:
                st.json(dados_receita)
        
        # Dados BrasilAPI
        if dados_brasilapi:
            with st.expander("🌐 Dados Enriquecidos - BrasilAPI"):
                
                # CNPJ
                if 'cnpj' in dados_brasilapi:
                    st.subheader("📊 Dados Cadastrais CNPJ")
                    cnpj_data = dados_brasilapi['cnpj']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Razão Social", cnpj_data.get('razao_social', 'N/A')[:30])
                        st.metric("Situação", cnpj_data.get('situacao_cadastral', 'N/A'))
                    with col2:
                        st.metric("CNAE", cnpj_data.get('cnae_principal', 'N/A'))
                        st.metric("Porte", cnpj_data.get('porte', 'N/A'))
                    with col3:
                        st.metric("UF", cnpj_data.get('uf', 'N/A'))
                        st.metric("Município", cnpj_data.get('municipio', 'N/A'))
                    
                    if cnpj_data.get('data_inicio_atividade'):
                        idade = calcular_idade_empresa(cnpj_data['data_inicio_atividade'])
                        if idade:
                            st.info(f"📅 Empresa em atividade há {idade:.1f} anos")
                    
                    with st.expander("Ver JSON completo"):
                        st.json(cnpj_data)
                
                # CEP
                if 'cep' in dados_brasilapi:
                    st.subheader("📍 Dados de Endereço")
                    cep_data = dados_brasilapi['cep']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Logradouro:** {cep_data.get('logradouro', 'N/A')}")
                        st.write(f"**Bairro:** {cep_data.get('bairro', 'N/A')}")
                        st.write(f"**Município:** {cep_data.get('municipio', 'N/A')}")
                        st.write(f"**UF:** {cep_data.get('uf', 'N/A')}")
                    with col2:
                        if cep_data.get('geo_disponivel'):
                            st.success("✅ Geolocalização disponível")
                            st.write(f"**Latitude:** {cep_data.get('latitude')}")
                            st.write(f"**Longitude:** {cep_data.get('longitude')}")
                        else:
                            st.warning("⚠️ Geolocalização não disponível")
                
                # FIPE
                if 'fipe' in dados_brasilapi:
                    st.subheader("🚗 Valor de Referência FIPE")
                    fipe_data = dados_brasilapi['fipe']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        valor = fipe_data.get('valor_formatado', 'N/A')
                        st.metric("Valor FIPE", valor)
                    with col2:
                        marca_modelo = f"{fipe_data.get('marca', '')} {fipe_data.get('modelo', '')}"
                        st.metric("Veículo", marca_modelo[:30])
                    with col3:
                        ano_comb = f"{fipe_data.get('ano_modelo', '')} / {fipe_data.get('combustivel', '')}"
                        st.metric("Ano/Combustível", ano_comb)
                    
                    st.info(f"📅 Referência: {fipe_data.get('mes_referencia', 'N/A')}")
        
        # Clientes Similares
        if similares:
            st.subheader("👥 Clientes Similares")
            df_similares = pd.DataFrame(similares)
            df_similares['similaridade'] = (df_similares['similaridade'] * 100).round(2)
            df_similares.columns = ['Nome', 'Score Histórico', 'Similaridade (%)', 'Sinistros 12m']
            st.dataframe(df_similares, use_container_width=True, hide_index=True)
        
        # Download JSON
        st.subheader("💾 Exportar Resultado")
        
        resultado_completo = {
            'cliente': {'cpf': cpf_limpo, 'nome': nome_input},
            'timestamp': datetime.now().isoformat(),
            'score': resultado,
            'dados_externos': dados_externos,
            'dados_brasilapi': dados_brasilapi,
            'similares': similares
        }
        
        st.download_button(
            "⬇️ Baixar JSON Completo",
            data=json.dumps(resultado_completo, indent=2, ensure_ascii=False),
            file_name=f"score_{cpf_limpo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
