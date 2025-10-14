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
                'logradouro': data.get('street'),
                'status': 'success'
            }
        return {'status': 'not_found'}
    except:
        return {'status': 'error'}

def consultar_fipe(marca: str, modelo: str):
    try:
        url_tabelas = f"{BASE_URL_BRASILAPI}/fipe/tabelas/v1"
        resp_tab = requests.get(url_tabelas, timeout=10)
        if resp_tab.status_code != 200:
            return {'status': 'error'}
        
        tabelas = resp_tab.json()
        tabela_ref = str(tabelas[-1]['codigo'])
        
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
            return {'status': 'not_found'}
        
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
            return {'status': 'not_found'}
        
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
        query_pt = f"{query} Brasil Portuguese"
        
        payload = {
            "api_key": api_key,
            "query": query_pt,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 5,
            "exclude_domains": ["facebook.com", "instagram.com", "twitter.com"]
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

def calcular_confiabilidade(resultado: Dict, dominios_confiaveis: list) -> Dict:
    if resultado.get('status') != 'success':
        return {'nivel': 'BAIXA', 'cor': 'red', 'emoji': '❌', 'motivo': 'Erro na consulta', 'fontes': '0/0'}
    
    results = resultado.get('results', [])
    
    if not results:
        return {'nivel': 'BAIXA', 'cor': 'red', 'emoji': '❌', 'motivo': 'Nenhuma fonte encontrada', 'fontes': '0/0'}
    
    fontes_confiaveis = 0
    total_fontes = len(results)
    
    for result in results:
        url = result.get('url', '').lower()
        if any(dominio in url for dominio in dominios_confiaveis):
            fontes_confiaveis += 1
    
    percentual = (fontes_confiaveis / total_fontes) * 100 if total_fontes > 0 else 0
    
    if percentual >= 60:
        return {
            'nivel': 'ALTA',
            'cor': 'green',
            'emoji': '✅',
            'motivo': 'Maioria de fontes oficiais',
            'fontes': f'{fontes_confiaveis}/{total_fontes} fontes confiáveis'
        }
    elif percentual >= 30:
        return {
            'nivel': 'MÉDIA',
            'cor': 'orange',
            'emoji': '⚠️',
            'motivo': 'Algumas fontes oficiais',
            'fontes': f'{fontes_confiaveis}/{total_fontes} fontes confiáveis'
        }
    else:
        return {
            'nivel': 'BAIXA',
            'cor': 'red',
            'emoji': '❌',
            'motivo': 'Poucas fontes oficiais',
            'fontes': f'{fontes_confiaveis}/{total_fontes} fontes confiáveis'
        }

# ================================
# ANÁLISES TAVILY - VEICULARES
# ================================
def analisar_veiculo_tavily(marca: str, modelo: str, ano: str, api_key: str, tipo: str):
    queries = {
        'recalls': f"recall {marca} {modelo} {ano} Procon defeitos",
        'custo': f"custo manutenção {marca} {modelo} preço peças",
        'seguranca': f"Latin NCAP {marca} {modelo} {ano} crash test estrelas",
        'roubos': f"ranking veículos roubados 2024 {marca} {modelo}"
    }
    
    dominios = {
        'recalls': ['procon.', '.gov.br', 'inmetro.gov.br'],
        'custo': ['quatrorodas.com', 'autoesporte.com'],
        'seguranca': ['latinncap.com', 'autoesporte.com'],
        'roubos': ['.gov.br', 'ssp.', 'policia']
    }
    
    resultado = consultar_tavily(queries[tipo], api_key)
    confiabilidade = calcular_confiabilidade(resultado, dominios[tipo])
    
    if resultado.get('status') != 'success':
        return {'ajuste': 0, 'reasons': [], 'resumo': '', 'confiabilidade': confiabilidade}
    
    answer = resultado.get('answer', '').lower()
    ajuste = 0
    reasons = []
    
    if confiabilidade['nivel'] in ['ALTA', 'MÉDIA']:
        if tipo == 'recalls':
            if 'recall crítico' in answer or 'defeito grave' in answer:
                ajuste = -8
                reasons.append(f"{marca} {modelo} com recall crítico (-8 pts)")
            elif 'recall' in answer:
                ajuste = -3
                reasons.append(f"{marca} {modelo} possui recall ativo (-3 pts)")
        
        elif tipo == 'custo':
            if 'custo elevado' in answer or 'caro' in answer:
                ajuste = -5
                reasons.append(f"{marca} {modelo} - alto custo de manutenção (-5 pts)")
            elif 'econômico' in answer or 'barato' in answer:
                ajuste = 2
                reasons.append(f"{marca} {modelo} - manutenção econômica (+2 pts)")
        
        elif tipo == 'seguranca':
            if '5 estrelas' in answer:
                ajuste = 5
                reasons.append(f"{marca} {modelo} - 5 estrelas Latin NCAP (+5 pts)")
            elif '4 estrelas' in answer:
                ajuste = 3
                reasons.append(f"{marca} {modelo} - 4 estrelas Latin NCAP (+3 pts)")
            elif '2 estrelas' in answer or '1 estrela' in answer:
                ajuste = -5
                reasons.append(f"{marca} {modelo} - baixa segurança (-5 pts)")
        
        elif tipo == 'roubos':
            if modelo.lower() in answer:
                if 'top 5' in answer or 'mais roubado' in answer:
                    ajuste = -10
                    reasons.append(f"{marca} {modelo} entre os MAIS roubados (-10 pts)")
                elif 'top 10' in answer:
                    ajuste = -5
                    reasons.append(f"{marca} {modelo} em ranking de roubos (-5 pts)")
    
    return {
        'ajuste': ajuste,
        'reasons': reasons,
        'resumo': resultado.get('answer', '')[:250],
        'confiabilidade': confiabilidade
    }

# ================================
# ANÁLISES TAVILY - REGIONAIS
# ================================
def analisar_regiao_tavily(municipio: str, uf: str, api_key: str, tipo: str):
    queries = {
        'acidentes': f"estatísticas acidentes trânsito {municipio} {uf} 2024 DETRAN",
        'vias': f"condição estradas {municipio} {uf} buracos pavimentação",
        'fiscalizacao': f"radares fiscalização trânsito {municipio} {uf}",
        'criminalidade': f"taxa roubo veículos {municipio} {uf} 2024",
        'frota': f"frota veículos {municipio} {uf} DETRAN densidade"
    }
    
    dominios = {
        'acidentes': ['detran.', '.gov.br', 'dnit.gov.br'],
        'vias': ['dnit.gov.br', 'cnt.org.br', '.gov.br'],
        'fiscalizacao': ['detran.', 'policia', '.gov.br'],
        'criminalidade': ['.gov.br', 'ssp.', 'policia'],
        'frota': ['detran.', '.gov.br', 'denatran.gov.br']
    }
    
    resultado = consultar_tavily(queries[tipo], api_key)
    confiabilidade = calcular_confiabilidade(resultado, dominios[tipo])
    
    if resultado.get('status') != 'success':
        return {'ajuste': 0, 'reasons': [], 'resumo': '', 'confiabilidade': confiabilidade}
    
    answer = resultado.get('answer', '').lower()
    ajuste = 0
    reasons = []
    
    if confiabilidade['nivel'] in ['ALTA', 'MÉDIA']:
        if tipo == 'acidentes':
            if 'alto índice' in answer or 'muitos acidentes' in answer:
                ajuste = -10
                reasons.append(f"{municipio}/{uf} - alto índice de acidentes (-10 pts)")
            elif 'moderado' in answer:
                ajuste = -5
                reasons.append(f"{municipio}/{uf} - índice moderado de acidentes (-5 pts)")
        
        elif tipo == 'vias':
            if 'péssima' in answer or 'buracos' in answer:
                ajuste = -6
                reasons.append(f"{municipio}/{uf} - vias em más condições (-6 pts)")
            elif 'regular' in answer:
                ajuste = -3
                reasons.append(f"{municipio}/{uf} - infraestrutura regular (-3 pts)")
        
        elif tipo == 'fiscalizacao':
            if 'intensa fiscalização' in answer:
                ajuste = 4
                reasons.append(f"{municipio}/{uf} - fiscalização intensa (+4 pts)")
            elif 'pouca fiscalização' in answer:
                ajuste = -2
                reasons.append(f"{municipio}/{uf} - fiscalização deficiente (-2 pts)")
        
        elif tipo == 'criminalidade':
            if 'alto índice' in answer or 'crítico' in answer:
                ajuste = -8
                reasons.append(f"{municipio}/{uf} - alto índice de roubo de veículos (-8 pts)")
            elif 'moderado' in answer:
                ajuste = -5
                reasons.append(f"{municipio}/{uf} - criminalidade moderada (-5 pts)")
        
        elif tipo == 'frota':
            if 'alta densidade' in answer or 'congestionamento' in answer:
                ajuste = -5
                reasons.append(f"{municipio}/{uf} - alta densidade de veículos (-5 pts)")
    
    return {
        'ajuste': ajuste,
        'reasons': reasons,
        'resumo': resultado.get('answer', '')[:250],
        'confiabilidade': confiabilidade
    }

# ================================
# AJUSTES BRASILAPI
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
    
    # Formulário
    st.header("📋 Dados para Análise")
    
    cep_input = st.text_input("CEP", placeholder="00000-000")
    cnpj_input = st.text_input("CNPJ Empregador (Opcional)", placeholder="00.000.000/0000-00")
    
    st.subheader("🚗 Dados do Veículo")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        marca_input = st.text_input("Marca", placeholder="Ex: Volkswagen")
    with col2:
        modelo_input = st.text_input("Modelo", placeholder="Ex: Gol")
    with col3:
        ano_input = st.text_input("Ano", placeholder="Ex: 2020")
    
    # Botão
    if st.button("🚀 Analisar Risco", type="primary", use_container_width=True):
        
        if not cep_input:
            st.error("⚠️ Preencha o CEP")
            return
        
        with st.spinner("🔄 Processando análise..."):
            progress_bar = st.progress(0)
            
            score_base = 70.0
            ajuste_total = 0
            todas_reasons = []
            dados_brasilapi = {}
            insights_tavily = []
            
            # CEP
            st.info("📍 Consultando CEP...")
            progress_bar.progress(20)
            
            dados_cep = consultar_cep(cep_input)
            if dados_cep.get('status') == 'success':
                dados_brasilapi['cep'] = dados_cep
            
            # CNPJ
            if cnpj_input:
                st.info("🏢 Consultando CNPJ...")
                progress_bar.progress(30)
                
                dados_cnpj = consultar_cnpj(cnpj_input)
                if dados_cnpj.get('status') == 'success':
                    dados_brasilapi['cnpj'] = dados_cnpj
                    ajuste_cnpj = calcular_ajuste_cnpj(dados_cnpj)
                    ajuste_total += ajuste_cnpj['ajuste']
                    todas_reasons.extend(ajuste_cnpj['reasons'])
            
            # FIPE
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
            
            # TAVILY
            tavily_key = st.secrets.get("TAVILY_API_KEY")
            
            if tavily_key:
                st.info("🧠 Executando análises Tavily...")
                
                # Análises Veiculares
                if marca_input and modelo_input:
                    ano = ano_input if ano_input else '2020'
                    
                    tipos_veiculo = [
                        ('recalls', '🔧 Recalls'),
                        ('custo', '💰 Custo Manutenção'),
                        ('seguranca', '🛡️ Segurança'),
                        ('roubos', '🚨 Ranking Roubos')
                    ]
                    
                    for idx, (tipo, nome) in enumerate(tipos_veiculo):
                        st.caption(f"Analisando {nome.lower()}...")
                        analise = analisar_veiculo_tavily(marca_input, modelo_input, ano, tavily_key, tipo)
                        ajuste_total += analise.get('ajuste', 0)
                        todas_reasons.extend(analise.get('reasons', []))
                        if analise.get('resumo'):
                            insights_tavily.append({
                                'tipo': nome,
                                'texto': analise['resumo'],
                                'confiabilidade': analise.get('confiabilidade', {})
                            })
                        progress_bar.progress(50 + (idx + 1) * 3)
                
                # Análises Regionais
                if dados_cep.get('status') == 'success':
                    municipio = dados_cep.get('municipio', '')
                    uf = dados_cep.get('uf', '')
                    
                    tipos_regiao = [
                        ('acidentes', '🚗 Acidentes Trânsito'),
                        ('vias', '🛣️ Qualidade das Vias'),
                        ('fiscalizacao', '🚔 Fiscalização'),
                        ('criminalidade', '⚠️ Criminalidade'),
                        ('frota', '🚙 Densidade de Frota')
                    ]
                    
                    for idx, (tipo, nome) in enumerate(tipos_regiao):
                        st.caption(f"Analisando {nome.lower()}...")
                        analise = analisar_regiao_tavily(municipio, uf, tavily_key, tipo)
                        ajuste_total += analise.get('ajuste', 0)
                        todas_reasons.extend(analise.get('reasons', []))
                        if analise.get('resumo'):
                            insights_tavily.append({
                                'tipo': nome,
                                'texto': analise['resumo'],
                                'confiabilidade': analise.get('confiabilidade', {})
                            })
                        progress_bar.progress(65 + (idx + 1) * 5)
            
            progress_bar.progress(100)
            
            # Calcula score final
            score_final = max(0, min(100, score_base + ajuste_total))
            
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
        
        # Fatores
        if todas_reasons:
            st.subheader("🎯 Fatores de Impacto")
            for i, reason in enumerate(todas_reasons, 1):
                st.write(f"{i}. {reason}")
        
        # Insights Tavily
        if insights_tavily:
            st.subheader("🧠 Insights Tavily Intelligence")
            
            st.info("""
            **ℹ️ Sobre a Confiabilidade:**
            - ✅ **ALTA**: Maioria das fontes são oficiais
            - ⚠️ **MÉDIA**: Algumas fontes oficiais
            - ❌ **BAIXA**: Poucas fontes oficiais
            """)
            
            for insight in insights_tavily:
                conf = insight.get('confiabilidade', {
                    'nivel': 'MÉDIA', 'cor': 'orange', 'emoji': '⚠️',
                    'motivo': 'N/A', 'fontes': 'N/A'
                })
                
                col_header, col_selo = st.columns([4, 1])
                
                with col_header:
                    st.markdown(f"### {insight.get('tipo', 'Análise')}")
                
                with col_selo:
                    if conf.get('nivel') == 'ALTA':
                        st.success(f"{conf.get('emoji', '✅')} ALTA")
                    elif conf.get('nivel') == 'MÉDIA':
                        st.warning(f"{conf.get('emoji', '⚠️')} MÉDIA")
                    else:
                        st.error(f"{conf.get('emoji', '❌')} BAIXA")
                
                st.info(insight.get('texto', 'Sem informações'))
                
                with st.expander("📊 Detalhes de Confiabilidade"):
                    st.write(f"**Nível:** {conf.get('nivel', 'N/A')}")
                    st.write(f"**Motivo:** {conf.get('motivo', 'N/A')}")
                    st.write(f"**Fontes:** {conf.get('fontes', 'N/A')}")
                
                st.markdown("---")
        
        # Dados BrasilAPI
        if dados_brasilapi:
            with st.expander("🌐 Dados BrasilAPI"):
                st.json(dados_brasilapi)
        
        # Download
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
