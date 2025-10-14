import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, Optional
import re

# ================================
# CONFIGURAÇÃO
# ================================
st.set_page_config(
    page_title="Sistema de Score de Risco",
    page_icon="🛡️",
    layout="wide"
)

BASE_URL_BRASILAPI = "https://brasilapi.com.br/api"
TAVILY_API_URL = "https://api.tavily.com/search"

# ================================
# FUNÇÕES AUXILIARES
# ================================
def normalizar_cnpj(cnpj: str) -> str:
    return re.sub(r'\D', '', cnpj)

def normalizar_cep(cep: str) -> str:
    return re.sub(r'\D', '', cep)

def parse_valor_brl(valor_str: str) -> float:
    if not valor_str:
        return 0.0
    valor_limpo = valor_str.replace('R$', '').replace(' ', '').replace('.', '').replace(',', '.')
    try:
        return float(valor_limpo)
    except:
        return 0.0

def calcular_idade_empresa(data_inicio: str):
    try:
        inicio = datetime.strptime(data_inicio, '%Y-%m-%d')
        hoje = datetime.now()
        delta = hoje - inicio
        return round(delta.days / 365.25, 2)
    except:
        return None

# ================================
# BRASILAPI
# ================================
def consultar_cnpj(cnpj: str):
    try:
        cnpj_limpo = normalizar_cnpj(cnpj)
        url = f"{BASE_URL_BRASILAPI}/cnpj/v1/{cnpj_limpo}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'cnpj': data.get('cnpj'),
                'razao_social': data.get('razao_social'),
                'situacao_cadastral': data.get('descricao_situacao_cadastral'),
                'data_inicio_atividade': data.get('data_inicio_atividade'),
                'cnae_principal': data.get('cnae_fiscal'),
                'cnae_descricao': data.get('cnae_fiscal_descricao'),
                'porte': data.get('porte'),
                'uf': data.get('uf'),
                'municipio': data.get('municipio'),
                'status': 'success'
            }
        return {'status': 'not_found'}
    except:
        return {'status': 'error'}

def consultar_cep(cep: str):
    try:
        cep_limpo = normalizar_cep(cep)
        url = f"{BASE_URL_BRASILAPI}/cep/v2/{cep_limpo}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'cep': data.get('cep'),
                'uf': data.get('state'),
                'municipio': data.get('city'),
                'bairro': data.get('neighborhood'),
                'status': 'success'
            }
        return {'status': 'not_found'}
    except:
        return {'status': 'error'}

def consultar_fipe(marca: str, modelo: str):
    try:
        # Busca tabela atual
        url_tabelas = f"{BASE_URL_BRASILAPI}/fipe/tabelas/v1"
        resp_tab = requests.get(url_tabelas, timeout=10)
        if resp_tab.status_code != 200:
            return {'status': 'error'}
        
        tabelas = resp_tab.json()
        tabela_ref = str(tabelas[-1]['codigo'])
        
        # Busca marcas
        url_marcas = f"{BASE_URL_BRASILAPI}/fipe/marcas/v1/carros"
        resp_marcas = requests.get(url_marcas, params={'tabela_referencia': tabela_ref}, timeout=10)
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
        url_modelos = f"{BASE_URL_BRASILAPI}/fipe/marcas/{codigo_marca}/modelos"
        resp_mod = requests.get(url_modelos, params={'tabela_referencia': tabela_ref}, timeout=10)
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
        resp_preco = requests.get(url_preco, params={'tabela_referencia': tabela_ref}, timeout=10)
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
            'status': 'success'
        }
    except:
        return {'status': 'error'}

# ================================
# TAVILY API
# ================================
def consultar_tavily(query: str, api_key: str) -> Optional[Dict]:
    try:
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 3
        }
        
        response = requests.post(TAVILY_API_URL, json=payload, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'answer': data.get('answer', ''),
                'results': data.get('results', []),
                'status': 'success'
            }
        return {'status': 'error'}
    except:
        return {'status': 'error'}

# ================================
# ANÁLISES TAVILY
# ================================
def analisar_veiculo_recalls(marca: str, modelo: str, ano: str, api_key: str) -> Dict:
    query = f"recall {marca} {modelo} {ano} Brasil Procon defeitos problemas"
    resultado = consultar_tavily(query, api_key)
    
    if resultado.get('status') != 'success':
        return {'ajuste': 0, 'reasons': [], 'resumo': ''}
    
    answer = resultado.get('answer', '').lower()
    ajuste = 0
    reasons = []
    
    if any(palavra in answer for palavra in ['recall crítico', 'defeito grave', 'risco']):
        ajuste = -8
        reasons.append(f"{marca} {modelo} com recall crítico (-8 pts)")
    elif 'recall' in answer:
        ajuste = -3
        reasons.append(f"{marca} {modelo} possui recall ativo (-3 pts)")
    
    return {'ajuste': ajuste, 'reasons': reasons, 'resumo': resultado.get('answer', '')[:250]}

def analisar_veiculo_seguranca(marca: str, modelo: str, ano: str, api_key: str) -> Dict:
    query = f"Latin NCAP {marca} {modelo} {ano} crash test estrelas segurança"
    resultado = consultar_tavily(query, api_key)
    
    if resultado.get('status') != 'success':
        return {'ajuste': 0, 'reasons': [], 'resumo': ''}
    
    answer = resultado.get('answer', '').lower()
    ajuste = 0
    reasons = []
    
    if '5 estrelas' in answer:
        ajuste = 5
        reasons.append(f"{marca} {modelo} - 5 estrelas Latin NCAP (+5 pts)")
    elif '4 estrelas' in answer:
        ajuste = 3
        reasons.append(f"{marca} {modelo} - 4 estrelas Latin NCAP (+3 pts)")
    elif any(palavra in answer for palavra in ['2 estrelas', '1 estrela']):
        ajuste = -5
        reasons.append(f"{marca} {modelo} - baixa avaliação de segurança (-5 pts)")
    
    return {'ajuste': ajuste, 'reasons': reasons, 'resumo': resultado.get('answer', '')[:250]}

def analisar_veiculo_roubado(marca: str, modelo: str, api_key: str) -> Dict:
    query = f"ranking veículos mais roubados Brasil 2024 {marca} {modelo}"
    resultado = consultar_tavily(query, api_key)
    
    if resultado.get('status') != 'success':
        return {'ajuste': 0, 'reasons': [], 'resumo': ''}
    
    answer = resultado.get('answer', '').lower()
    modelo_lower = modelo.lower()
    ajuste = 0
    reasons = []
    
    if modelo_lower in answer:
        if any(palavra in answer for palavra in ['primeiro', 'top 5', 'mais roubado']):
            ajuste = -10
            reasons.append(f"{marca} {modelo} entre os MAIS roubados (-10 pts)")
        elif 'top 10' in answer:
            ajuste = -5
            reasons.append(f"{marca} {modelo} em ranking de roubos (-5 pts)")
    
    return {'ajuste': ajuste, 'reasons': reasons, 'resumo': resultado.get('answer', '')[:250]}

def analisar_acidentes_regiao(municipio: str, uf: str, api_key: str) -> Dict:
    query = f"estatísticas acidentes trânsito {municipio} {uf} 2024 DETRAN"
    resultado = consultar_tavily(query, api_key)
    
    if resultado.get('status') != 'success':
        return {'ajuste': 0, 'reasons': [], 'resumo': ''}
    
    answer = resultado.get('answer', '').lower()
    ajuste = 0
    reasons = []
    
    if any(palavra in answer for palavra in ['alto índice', 'muitos acidentes']):
        ajuste = -10
        reasons.append(f"{municipio}/{uf} - alto índice de acidentes (-10 pts)")
    elif 'moderado' in answer:
        ajuste = -5
        reasons.append(f"{municipio}/{uf} - índice moderado de acidentes (-5 pts)")
    
    return {'ajuste': ajuste, 'reasons': reasons, 'resumo': resultado.get('answer', '')[:250]}

def analisar_criminalidade_regiao(municipio: str, uf: str, api_key: str) -> Dict:
    query = f"taxa roubo veículos {municipio} {uf} Brasil estatísticas 2024"
    resultado = consultar_tavily(query, api_key)
    
    if resultado.get('status') != 'success':
        return {'ajuste': 0, 'reasons': [], 'resumo': ''}
    
    answer = resultado.get('answer', '').lower()
    ajuste = 0
    reasons = []
    
    if any(palavra in answer for palavra in ['alto índice', 'elevado', 'crítico']):
        ajuste = -8
        reasons.append(f"{municipio}/{uf} - alto índice de roubo de veículos (-8 pts)")
    elif 'moderado' in answer:
        ajuste = -5
        reasons.append(f"{municipio}/{uf} - índice moderado de criminalidade (-5 pts)")
    
    return {'ajuste': ajuste, 'reasons': reasons, 'resumo': resultado.get('answer', '')[:250]}

def analisar_saude_empresa(razao_social: str, cnpj: str, api_key: str) -> Dict:
    query = f"{razao_social} CNPJ {cnpj} falência recuperação judicial 2024"
    resultado = consultar_tavily(query, api_key)
    
    if resultado.get('status') != 'success':
        return {'ajuste': 0, 'reasons': [], 'resumo': ''}
    
    answer = resultado.get('answer', '').lower()
    ajuste = 0
    reasons = []
    
    if any(palavra in answer for palavra in ['falência', 'recuperação judicial']):
        ajuste = -10
        reasons.append(f"Empresa em situação financeira crítica (-10 pts)")
    elif 'dívidas' in answer:
        ajuste = -5
        reasons.append(f"Empresa com dificuldades financeiras (-5 pts)")
    
    return {'ajuste': ajuste, 'reasons': reasons, 'resumo': resultado.get('answer', '')[:250]}

# ================================
# CÁLCULO DE AJUSTES BRASILAPI
# ================================
def calcular_ajuste_cnpj(dados_cnpj):
    if dados_cnpj.get('status') != 'success':
        return {'ajuste': 0, 'reasons': []}
    
    ajuste = 0
    reasons = []
    
    situacao = dados_cnpj.get('situacao_cadastral', '')
    if 'ATIVA' in situacao.upper():
        idade = calcular_idade_empresa(dados_cnpj.get('data_inicio_atividade', ''))
        if idade and idade >= 10:
            ajuste += 5
            reasons.append(f"Empresa ativa há {idade:.1f} anos (+5 pts)")
        elif idade and idade >= 5:
            ajuste += 3
            reasons.append(f"Empresa ativa há {idade:.1f} anos (+3 pts)")
    else:
        ajuste -= 10
        reasons.append(f"Empresa: {situacao} (-10 pts)")
    
    return {'ajuste': ajuste, 'reasons': reasons}

def calcular_ajuste_fipe(dados_fipe):
    if dados_fipe.get('status') != 'success':
        return {'ajuste': 0, 'reasons': []}
    
    ajuste = 0
    reasons = []
    valor = dados_fipe.get('valor_numerico', 0)
    
    if valor >= 100000:
        ajuste -= 8
        reasons.append(f"Veículo alto valor: R$ {valor:,.2f} (-8 pts)")
    elif valor >= 60000:
        ajuste -= 5
        reasons.append(f"Veículo valor elevado: R$ {valor:,.2f} (-5 pts)")
    elif valor >= 30000:
        ajuste -= 2
        reasons.append(f"Veículo valor médio: R$ {valor:,.2f} (-2 pts)")
    
    return {'ajuste': ajuste, 'reasons': reasons}

# ================================
# INTERFACE STREAMLIT
# ================================
def main():
    st.title("🛡️ Sistema de Score de Risco")
    st.markdown("**Análise inteligente com Tavily + BrasilAPI**")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Informações")
        st.markdown("""
        **APIs Utilizadas:**
        - 🌐 BrasilAPI (Pública)
        - 🧠 Tavily Intelligence
        
        **Status:**
        """)
        
        st.success("✅ BrasilAPI")
        
        tavily_key = st.secrets.get("TAVILY_API_KEY", None)
        if tavily_key:
            st.success("✅ Tavily API")
        else:
            st.warning("⚠️ Tavily não configurada")
    
    # Formulário Principal
    st.header("📋 Dados para Análise")
    
    # CEP
    cep_input = st.text_input("CEP", placeholder="00000-000", help="Para análise de risco regional")
    
    # CNPJ
    cnpj_input = st.text_input("CNPJ (Opcional)", placeholder="00.000.000/0000-00", help="Análise empresarial")
    
    # Veículo
    st.subheader("🚗 Dados do Veículo")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        marca_input = st.text_input("Marca", placeholder="Ex: Volkswagen")
    with col2:
        modelo_input = st.text_input("Modelo", placeholder="Ex: Gol")
    with col3:
        ano_input = st.text_input("Ano", placeholder="Ex: 2020")
    
    # Botão de análise
    if st.button("🚀 Analisar Risco", type="primary", use_container_width=True):
        
        if not cep_input:
            st.error("⚠️ Preencha ao menos o CEP")
            return
        
        with st.spinner("🔄 Processando análise..."):
            progress_bar = st.progress(0)
            
            # Inicializa resultados
            score_base = 70.0
            ajuste_total = 0
            todas_reasons = []
            dados_brasilapi = {}
            insights_tavily = []
            
            # 1. CEP
            st.info("📍 Consultando CEP...")
            progress_bar.progress(20)
            
            dados_cep = consultar_cep(cep_input)
            if dados_cep.get('status') == 'success':
                dados_brasilapi['cep'] = dados_cep
            
            # 2. CNPJ
            if cnpj_input:
                st.info("🏢 Consultando CNPJ...")
                progress_bar.progress(30)
                
                dados_cnpj = consultar_cnpj(cnpj_input)
                if dados_cnpj.get('status') == 'success':
                    dados_brasilapi['cnpj'] = dados_cnpj
                    ajuste_cnpj = calcular_ajuste_cnpj(dados_cnpj)
                    ajuste_total += ajuste_cnpj['ajuste']
                    todas_reasons.extend(ajuste_cnpj['reasons'])
            
            # 3. FIPE
            if marca_input and modelo_input:
                st.info("🚗 Consultando FIPE...")
                progress_bar.progress(40)
                
                dados_fipe = consultar_fipe(marca_input, modelo_input)
                if dados_fipe.get('status') == 'success':
                    dados_brasilapi['fipe'] = dados_fipe
                    ajuste_fipe = calcular_ajuste_fipe(dados_fipe)
                    ajuste_total += ajuste_fipe['ajuste']
                    todas_reasons.extend(ajuste_fipe['reasons'])
            
            progress_bar.progress(50)
            
            # 4. TAVILY
            tavily_key = st.secrets.get("TAVILY_API_KEY")
            
            if tavily_key:
                st.info("🧠 Executando análises Tavily...")
                
                # Análises Veiculares
                if marca_input and modelo_input:
                    ano = ano_input if ano_input else '2020'
                    
                    # Recalls
                    st.caption("🔧 Analisando recalls...")
                    analise = analisar_veiculo_recalls(marca_input, modelo_input, ano, tavily_key)
                    if analise['ajuste'] != 0:
                        ajuste_total += analise['ajuste']
                        todas_reasons.extend(analise['reasons'])
                        if analise['resumo']:
                            insights_tavily.append({'tipo': '🔧 Recalls', 'texto': analise['resumo']})
                    
                    progress_bar.progress(60)
                    
                    # Segurança
                    st.caption("🛡️ Analisando segurança...")
                    analise = analisar_veiculo_seguranca(marca_input, modelo_input, ano, tavily_key)
                    if analise['ajuste'] != 0:
                        ajuste_total += analise['ajuste']
                        todas_reasons.extend(analise['reasons'])
                        if analise['resumo']:
                            insights_tavily.append({'tipo': '🛡️ Segurança', 'texto': analise['resumo']})
                    
                    progress_bar.progress(70)
                    
                    # Roubos
                    st.caption("🚨 Verificando ranking de roubos...")
                    analise = analisar_veiculo_roubado(marca_input, modelo_input, tavily_key)
                    if analise['ajuste'] != 0:
                        ajuste_total += analise['ajuste']
                        todas_reasons.extend(analise['reasons'])
                        if analise['resumo']:
                            insights_tavily.append({'tipo': '🚨 Ranking Roubos', 'texto': analise['resumo']})
                
                # Análises Regionais
                if dados_cep.get('status') == 'success':
                    municipio = dados_cep.get('municipio', '')
                    uf = dados_cep.get('uf', '')
                    
                    progress_bar.progress(80)
                    
                    # Acidentes
                    st.caption("🚗 Analisando acidentes...")
                    analise = analisar_acidentes_regiao(municipio, uf, tavily_key)
                    if analise['ajuste'] != 0:
                        ajuste_total += analise['ajuste']
                        todas_reasons.extend(analise['reasons'])
                        if analise['resumo']:
                            insights_tavily.append({'tipo': '🚗 Acidentes Trânsito', 'texto': analise['resumo']})
                    
                    progress_bar.progress(85)
                    
                    # Criminalidade
                    st.caption("⚠️ Analisando criminalidade...")
                    analise = analisar_criminalidade_regiao(municipio, uf, tavily_key)
                    if analise['ajuste'] != 0:
                        ajuste_total += analise['ajuste']
                        todas_reasons.extend(analise['reasons'])
                        if analise['resumo']:
                            insights_tavily.append({'tipo': '⚠️ Criminalidade', 'texto': analise['resumo']})
                
                # Análise Empresarial
                if cnpj_input and dados_cnpj.get('status') == 'success':
                    progress_bar.progress(90)
                    
                    st.caption("💼 Analisando empresa...")
                    analise = analisar_saude_empresa(
                        dados_cnpj.get('razao_social', ''),
                        dados_cnpj.get('cnpj', ''),
                        tavily_key
                    )
                    if analise['ajuste'] != 0:
                        ajuste_total += analise['ajuste']
                        todas_reasons.extend(analise['reasons'])
                        if analise['resumo']:
                            insights_tavily.append({'tipo': '💼 Saúde Financeira', 'texto': analise['resumo']})
            
            progress_bar.progress(100)
            
            # Calcula score final
            score_final = max(0, min(100, score_base + ajuste_total))
            
            # Define banda
            if score_final >= 80:
                banda = 'MUITO BAIXO'
            elif score_final >= 60:
                banda = 'BAIXO'
            elif score_final >= 40:
                banda = 'MÉDIO'
            elif score_final >= 20:
                banda = 'ALTO'
            else:
                banda = 'MUITO ALTO'
        
        st.success("✅ Análise concluída!")
        
        # RESULTADOS
        st.header("📊 Resultado da Análise")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Score de Risco", f"{score_final:.1f}/100")
        
        with col2:
            st.metric("Banda de Risco", banda)
        
        with col3:
            cor = "🟢" if score_final >= 70 else "🟡" if score_final >= 40 else "🔴"
            st.metric("Status", cor)
        
        # Fatores de Impacto
        if todas_reasons:
            st.subheader("🎯 Fatores de Impacto")
            for i, reason in enumerate(todas_reasons, 1):
                st.write(f"{i}. {reason}")
        
        # Insights Tavily
        if insights_tavily:
            st.subheader("🧠 Insights Tavily Intelligence")
            for insight in insights_tavily:
                with st.expander(f"{insight['tipo']}", expanded=False):
                    st.info(insight['texto'])
        
        # Dados BrasilAPI
        if dados_brasilapi:
            with st.expander("🌐 Dados BrasilAPI"):
                st.json(dados_brasilapi)
        
        # Download JSON
        st.subheader("💾 Exportar")
        
        resultado_completo = {
            'timestamp': datetime.now().isoformat(),
            'score': score_final,
            'banda': banda,
            'ajuste_total': ajuste_total,
            'reasons': todas_reasons,
            'dados_brasilapi': dados_brasilapi,
            'insights_tavily': insights_tavily
        }
        
        st.download_button(
            "⬇️ Baixar JSON",
            data=json.dumps(resultado_completo, indent=2, ensure_ascii=False),
            file_name=f"score_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
