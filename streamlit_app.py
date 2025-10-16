"""
Módulo de Análise Avançada com Embeddings
Sistema de Score Duplo: APIs + Similaridade Vetorial
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import streamlit as st
from datetime import datetime, timedelta

class AnalisadorEmbeddings:
    """
    Análise de similaridade com base vetorizada
    Segunda camada do sistema de score
    """
    
    def __init__(self, caminho_pkl: str = None):
        self.base_embeddings = None
        self.scaler = StandardScaler()
        self.estatisticas_base = {}
        
        if caminho_pkl:
            self.carregar_base(caminho_pkl)
    
    def carregar_base(self, caminho_pkl: str):
        """Carrega base de embeddings do arquivo PKL"""
        try:
            with open(caminho_pkl, 'rb') as f:
                self.base_embeddings = pickle.load(f)
            
            # Calcular estatísticas da base
            self._calcular_estatisticas()
            return True
        except Exception as e:
            st.error(f"Erro ao carregar PKL: {e}")
            return False
    
    def _calcular_estatisticas(self):
        """Calcula estatísticas gerais da base para benchmarking"""
        if self.base_embeddings is None:
            return
        
        df = pd.DataFrame(self.base_embeddings)
        
        self.estatisticas_base = {
            'total_clientes': len(df),
            'media_sinistros_12m': df['historico_sinistros.total_sinistros_12m'].mean(),
            'mediana_sinistros_12m': df['historico_sinistros.total_sinistros_12m'].median(),
            'percentil_90_sinistros': df['historico_sinistros.total_sinistros_12m'].quantile(0.9),
            'valor_medio_sinistro': df['historico_sinistros.valor_medio_sinistro'].mean(),
            'taxa_sinistralidade': (df['historico_sinistros.total_sinistros_12m'] > 0).mean(),
            
            # Análise por veículo
            'veiculos_mais_sinistrados': df.groupby(['veiculo.marca', 'veiculo.modelo'])[
                'historico_sinistros.total_sinistros_12m'
            ].mean().nlargest(10).to_dict(),
            
            # Análise por região (CEP)
            'regioes_criticas': df.groupby(df['localizacao.cep'].str[:5])[
                'historico_sinistros.total_sinistros_12m'
            ].mean().nlargest(10).to_dict(),
            
            # Análise por tipo de sinistro
            'tipos_sinistro': df['historico_sinistros.tipo_predominante'].value_counts().to_dict(),
            
            # Peças mais substituídas
            'pecas_frequentes': self._analisar_pecas(df)
        }
    
    def _analisar_pecas(self, df: pd.DataFrame) -> Dict:
        """Analisa as peças mais frequentemente substituídas"""
        todas_pecas = []
        for pecas in df['historico_sinistros.pecas_substituidas_12m'].dropna():
            if isinstance(pecas, str):
                todas_pecas.extend([p.strip() for p in pecas.split(',')])
        
        from collections import Counter
        return dict(Counter(todas_pecas).most_common(10))
    
    def vetorizar_cliente(self, dados_cliente: Dict) -> np.ndarray:
        """
        Converte dados do cliente em vetor numérico para comparação
        """
        # Features numéricas principais
        features = []
        
        # CEP - converter primeiros 5 dígitos em número
        cep = dados_cliente.get('localizacao', {}).get('cep', '00000')
        features.append(int(cep[:5]))
        
        # Veículo
        veiculo = dados_cliente.get('veiculo', {})
        features.append(veiculo.get('valor_fipe', 0))
        features.append(veiculo.get('ano_fabricacao', 2020))
        features.append(veiculo.get('ano_modelo', 2021))
        
        # Categoria do veículo (one-hot simplificado)
        categorias = ['Passeio', 'SUV', 'Pickup', 'Moto', 'Caminhão']
        cat_veiculo = veiculo.get('categoria', 'Passeio')
        for cat in categorias:
            features.append(1 if cat == cat_veiculo else 0)
        
        # Combustível (one-hot)
        combustiveis = ['Flex', 'Gasolina', 'Diesel', 'Elétrico', 'Híbrido']
        comb_veiculo = veiculo.get('combustivel', 'Flex')
        for comb in combustiveis:
            features.append(1 if comb == comb_veiculo else 0)
        
        # Marca/Modelo (hash simples para encoding)
        marca_hash = hash(veiculo.get('marca', '')) % 1000
        modelo_hash = hash(veiculo.get('modelo', '')) % 1000
        features.append(marca_hash)
        features.append(modelo_hash)
        
        # Histórico (se disponível - para clientes existentes)
        historico = dados_cliente.get('historico_sinistros', {})
        features.append(historico.get('total_sinistros_12m', 0))
        features.append(historico.get('total_sinistros_24m', 0))
        features.append(historico.get('valor_total_sinistros_12m', 0))
        features.append(historico.get('frequencia_anual', 0))
        features.append(historico.get('dias_desde_ultimo_sinistro', 365))
        
        return np.array(features)
    
    def encontrar_similares(self, dados_cliente: Dict, k: int = 100) -> pd.DataFrame:
        """
        Encontra os K clientes mais similares na base
        """
        if self.base_embeddings is None:
            return pd.DataFrame()
        
        # Vetorizar cliente atual
        vetor_cliente = self.vetorizar_cliente(dados_cliente).reshape(1, -1)
        
        # Vetorizar toda a base (cache isso em produção)
        vetores_base = np.array([
            self.vetorizar_cliente(cliente) 
            for cliente in self.base_embeddings
        ])
        
        # Normalizar vetores
        vetor_cliente_norm = self.scaler.fit_transform(vetor_cliente)
        vetores_base_norm = self.scaler.transform(vetores_base)
        
        # Calcular similaridade
        similaridades = cosine_similarity(vetor_cliente_norm, vetores_base_norm)[0]
        
        # Pegar top K mais similares
        indices_similares = np.argsort(similaridades)[-k:][::-1]
        
        # Criar DataFrame com resultados
        clientes_similares = []
        for idx in indices_similares:
            cliente = self.base_embeddings[idx]
            clientes_similares.append({
                'similaridade': similaridades[idx],
                'sinistros_12m': cliente['historico_sinistros']['total_sinistros_12m'],
                'valor_sinistros_12m': cliente['historico_sinistros']['valor_total_sinistros_12m'],
                'tipo_sinistro': cliente['historico_sinistros']['tipo_predominante'],
                'marca': cliente['veiculo']['marca'],
                'modelo': cliente['veiculo']['modelo'],
                'cep': cliente['localizacao']['cep'][:5] + 'xxx',  # Privacy
                'valor_fipe': cliente['veiculo']['valor_fipe']
            })
        
        return pd.DataFrame(clientes_similares)
    
    def calcular_score_similaridade(self, dados_cliente: Dict) -> Dict:
        """
        Calcula score baseado em clientes similares
        Retorna score de 0-1000 e análise detalhada
        """
        # Encontrar similares
        similares = self.encontrar_similares(dados_cliente, k=100)
        
        if similares.empty:
            return {
                'score': 500,
                'confianca': 'BAIXA',
                'analise': 'Sem dados suficientes para análise'
            }
        
        # Filtrar apenas os muito similares (>80% similaridade)
        muito_similares = similares[similares['similaridade'] > 0.8]
        
        if len(muito_similares) < 10:
            confianca = 'BAIXA'
        elif len(muito_similares) < 30:
            confianca = 'MÉDIA'
        else:
            confianca = 'ALTA'
        
        # Calcular métricas dos similares
        media_sinistros = muito_similares['sinistros_12m'].mean()
        taxa_sinistralidade = (muito_similares['sinistros_12m'] > 0).mean()
        valor_medio_sinistro = muito_similares['valor_sinistros_12m'].mean()
        
        # Comparar com a base geral
        desvio_media = media_sinistros - self.estatisticas_base['media_sinistros_12m']
        
        # Calcular score (0-1000)
        # Base: 500
        score = 500
        
        # Ajustes baseados em sinistralidade
        if taxa_sinistralidade < 0.2:  # Menos de 20% tiveram sinistros
            score += 150
        elif taxa_sinistralidade < 0.4:
            score += 50
        elif taxa_sinistralidade > 0.6:
            score -= 100
        elif taxa_sinistralidade > 0.8:
            score -= 200
        
        # Ajuste por média de sinistros
        if media_sinistros < 0.5:
            score += 100
        elif media_sinistros < 1:
            score += 50
        elif media_sinistros > 2:
            score -= 100
        elif media_sinistros > 3:
            score -= 150
        
        # Ajuste por valor médio
        if valor_medio_sinistro < 5000:
            score += 50
        elif valor_medio_sinistro > 15000:
            score -= 100
        elif valor_medio_sinistro > 25000:
            score -= 150
        
        # Limitar score
        score = max(0, min(1000, score))
        
        # Análise detalhada
        analise = {
            'score': score,
            'confianca': confianca,
            'total_similares': len(muito_similares),
            'taxa_sinistralidade': f"{taxa_sinistralidade:.1%}",
            'media_sinistros': f"{media_sinistros:.2f}",
            'valor_medio': f"R$ {valor_medio_sinistro:,.2f}",
            
            # Comparação com base
            'vs_base': {
                'sinistros': 'ACIMA' if desvio_media > 0 else 'ABAIXO',
                'desvio': f"{abs(desvio_media):.2f}",
                'percentil': self._calcular_percentil(media_sinistros)
            },
            
            # Insights
            'insights': self._gerar_insights(muito_similares, dados_cliente)
        }
        
        return analise
    
    def _calcular_percentil(self, valor: float) -> int:
        """Calcula em que percentil o valor está na base"""
        if self.base_embeddings is None:
            return 50
        
        df = pd.DataFrame(self.base_embeddings)
        todos_valores = df['historico_sinistros.total_sinistros_12m'].values
        percentil = (todos_valores < valor).mean() * 100
        return int(percentil)
    
    def _gerar_insights(self, similares: pd.DataFrame, dados_cliente: Dict) -> List[str]:
        """Gera insights específicos baseados nos similares"""
        insights = []
        
        # Sinistros mais comuns
        tipo_mais_comum = similares['tipo_sinistro'].mode()[0] if not similares.empty else 'Colisão'
        insights.append(f"🚗 Tipo de sinistro mais comum em perfis similares: {tipo_mais_comum}")
        
        # Taxa de sinistralidade
        taxa = (similares['sinistros_12m'] > 0).mean()
        if taxa < 0.3:
            insights.append(f"✅ Apenas {taxa:.0%} dos clientes similares tiveram sinistros")
        else:
            insights.append(f"⚠️ {taxa:.0%} dos clientes similares tiveram sinistros")
        
        # Valor médio
        valor_medio = similares['valor_sinistros_12m'].mean()
        if valor_medio < 5000:
            insights.append(f"💰 Sinistros de baixo valor em média (R$ {valor_medio:,.0f})")
        elif valor_medio > 15000:
            insights.append(f"💸 Sinistros de alto valor em média (R$ {valor_medio:,.0f})")
        
        # Comparação regional
        mesma_regiao = similares[similares['cep'].str[:2] == dados_cliente.get('localizacao', {}).get('cep', '')[:2]]
        if not mesma_regiao.empty:
            taxa_regiao = (mesma_regiao['sinistros_12m'] > 0).mean()
            insights.append(f"📍 Na sua região: {taxa_regiao:.0%} de sinistralidade")
        
        return insights

def calcular_score_final_duplo(
    score_apis: int,
    score_similaridade: int,
    confianca_similaridade: str
) -> Tuple[int, str]:
    """
    Combina os dois scores em um score final
    
    Args:
        score_apis: Score das APIs públicas (0-1000)
        score_similaridade: Score baseado em similaridade (0-1000)
        confianca_similaridade: ALTA, MÉDIA ou BAIXA
    
    Returns:
        (score_final, metodo_usado)
    """
    
    # Pesos baseados na confiança
    if confianca_similaridade == 'ALTA':
        # 70% similaridade, 30% APIs
        score_final = int(score_similaridade * 0.7 + score_apis * 0.3)
        metodo = "70% Similaridade + 30% APIs"
    
    elif confianca_similaridade == 'MÉDIA':
        # 50% cada
        score_final = int(score_similaridade * 0.5 + score_apis * 0.5)
        metodo = "50% Similaridade + 50% APIs"
    
    else:  # BAIXA
        # 20% similaridade, 80% APIs
        score_final = int(score_similaridade * 0.2 + score_apis * 0.8)
        metodo = "20% Similaridade + 80% APIs"
    
    return score_final, metodo

# ================================
# INTERFACE STREAMLIT PARA ANÁLISE DUPLA
# ================================
def exibir_analise_dupla(
    analisador: AnalisadorEmbeddings,
    dados_cliente: Dict,
    score_apis: int
):
    """
    Exibe interface completa da análise dupla
    """
    st.header("🔬 Análise Avançada com Base Vetorizada")
    
    # Calcular score de similaridade
    with st.spinner("🧮 Analisando similaridade com 1M+ clientes..."):
        resultado_similaridade = analisador.calcular_score_similaridade(dados_cliente)
    
    # Score final combinado
    score_final, metodo = calcular_score_final_duplo(
        score_apis,
        resultado_similaridade['score'],
        resultado_similaridade['confianca']
    )
    
    # Exibir resultados
    st.subheader("📊 Resultado da Análise Dupla")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Score APIs",
            f"{score_apis}",
            "Análise tradicional"
        )
    
    with col2:
        st.metric(
            "Score Similaridade",
            f"{resultado_similaridade['score']}",
            f"Confiança: {resultado_similaridade['confianca']}"
        )
    
    with col3:
        st.metric(
            "SCORE FINAL",
            f"{score_final}",
            metodo
        )
    
    with col4:
        classificacao = (
            "PREMIUM" if score_final >= 800 else
            "EXCELENTE" if score_final >= 650 else
            "BOM" if score_final >= 500 else
            "REGULAR" if score_final >= 350 else
            "ATENÇÃO" if score_final >= 200 else
            "CRÍTICO"
        )
        st.metric(
            "Classificação Final",
            classificacao,
            "🏆" if score_final >= 800 else "⭐" if score_final >= 650 else "✅"
        )
    
    # Detalhamento da análise de similaridade
    with st.expander("🔍 Detalhes da Análise de Similaridade", expanded=True):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Estatísticas dos Similares")
            st.write(f"**Clientes analisados:** {resultado_similaridade['total_similares']}")
            st.write(f"**Taxa de sinistralidade:** {resultado_similaridade['taxa_sinistralidade']}")
            st.write(f"**Média de sinistros/ano:** {resultado_similaridade['media_sinistros']}")
            st.write(f"**Valor médio sinistro:** {resultado_similaridade['valor_medio']}")
        
        with col2:
            st.markdown("### 📊 Comparação com Base Geral")
            vs = resultado_similaridade['vs_base']
            st.write(f"**Posição:** {vs['sinistros']} da média")
            st.write(f"**Desvio:** {vs['desvio']} sinistros/ano")
            st.write(f"**Percentil:** {vs['percentil']}º")
            
            # Barra de percentil
            st.progress(vs['percentil'] / 100)
        
        # Insights
        st.markdown("### 💡 Insights Automáticos")
        for insight in resultado_similaridade['insights']:
            st.info(insight)
    
    # Visualização dos similares
    with st.expander("👥 Ver Clientes Similares (Top 10)"):
        similares = analisador.encontrar_similares(dados_cliente, k=10)
        
        # Preparar para exibição
        similares_display = similares[['similaridade', 'marca', 'modelo', 
                                      'sinistros_12m', 'valor_sinistros_12m', 'cep']]
        similares_display['similaridade'] = similares_display['similaridade'].apply(lambda x: f"{x:.1%}")
        similares_display['valor_sinistros_12m'] = similares_display['valor_sinistros_12m'].apply(lambda x: f"R$ {x:,.0f}")
        
        st.dataframe(similares_display, use_container_width=True, hide_index=True)
    
    # Estatísticas da base
    if st.checkbox("📊 Ver Estatísticas Gerais da Base"):
        st.markdown("### Estatísticas da Base Vetorizada")
        
        stats = analisador.estatisticas_base
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Clientes", f"{stats['total_clientes']:,}")
            st.metric("Taxa de Sinistralidade", f"{stats['taxa_sinistralidade']:.1%}")
        
        with col2:
            st.metric("Média Sinistros/Ano", f"{stats['media_sinistros_12m']:.2f}")
            st.metric("Valor Médio Sinistro", f"R$ {stats['valor_medio_sinistro']:,.0f}")
        
        with col3:
            st.metric("Mediana Sinistros", f"{stats['mediana_sinistros_12m']:.1f}")
            st.metric("Percentil 90", f"{stats['percentil_90_sinistros']:.1f}")
        
        # Top veículos sinistrados
        st.markdown("#### 🚗 Top 5 Veículos Mais Sinistrados")
        veiculos_top = list(stats['veiculos_mais_sinistrados'].items())[:5]
        for (marca, modelo), media in veiculos_top:
            st.write(f"• {marca} {modelo}: {media:.2f} sinistros/ano")
        
        # Tipos de sinistro
        st.markdown("#### 🔧 Distribuição por Tipo de Sinistro")
        for tipo, qtd in list(stats['tipos_sinistro'].items())[:5]:
            st.write(f"• {tipo}: {qtd} casos")
    
    return score_final, classificacao

# ================================
# EXEMPLO DE USO
# ================================
if __name__ == "__main__":
    
    # Simular dados de cliente
    cliente_exemplo = {
        "identificacao": {
            "cpf": "12345678901",
            "nome_completo": "João Silva Santos"
        },
        "localizacao": {
            "cep": "01310100"
        },
        "veiculo": {
            "marca": "Volkswagen",
            "modelo": "Gol 1.6 MSI",
            "ano_fabricacao": 2020,
            "ano_modelo": 2021,
            "combustivel": "Flex",
            "cor": "Branco",
            "categoria": "Passeio",
            "valor_fipe": 58000.00
        },
        "historico_sinistros": {
            "total_sinistros_12m": 0,
            "total_sinistros_24m": 0,
            "total_sinistros_36m": 0,
            "valor_total_sinistros_12m": 0,
            "valor_medio_sinistro": 0,
            "tipo_predominante": "Nenhum",
            "pecas_substituidas_12m": "",
            "categoria_peca_mais_trocada": "",
            "frequencia_anual": 0,
            "dias_desde_ultimo_sinistro": 999
        }
    }
    
    # Criar analisador
    analisador = AnalisadorEmbeddings()
    
    # Carregar base (quando disponível)
    # analisador.carregar_base('clientes_embeddings.pkl')
    
    # Calcular score
    resultado = analisador.calcular_score_similaridade(cliente_exemplo)
    
    print(f"Score de Similaridade: {resultado['score']}")
    print(f"Confiança: {resultado['confianca']}")
    print(f"Insights: {resultado['insights']}")

# ================================
# FUNÇÕES UTILITÁRIAS ADICIONAIS
# ================================

def gerar_relatorio_completo(
    analisador: AnalisadorEmbeddings,
    dados_cliente: Dict,
    score_apis: int,
    output_path: str = None
) -> Dict:
    """
    Gera relatório completo da análise dupla
    """
    # Análise de similaridade
    resultado_similaridade = analisador.calcular_score_similaridade(dados_cliente)
    
    # Score final
    score_final, metodo = calcular_score_final_duplo(
        score_apis,
        resultado_similaridade['score'],
        resultado_similaridade['confianca']
    )
    
    # Classificação
    classificacao = (
        "PREMIUM" if score_final >= 800 else
        "EXCELENTE" if score_final >= 650 else
        "BOM" if score_final >= 500 else
        "REGULAR" if score_final >= 350 else
        "ATENÇÃO" if score_final >= 200 else
        "CRÍTICO"
    )
    
    # Montar relatório
    relatorio = {
        'timestamp': datetime.now().isoformat(),
        'dados_cliente': dados_cliente,
        'analise_apis': {
            'score': score_apis,
            'fonte': 'BrasilAPI + Tavily'
        },
        'analise_similaridade': resultado_similaridade,
        'score_final': {
            'valor': score_final,
            'metodo_calculo': metodo,
            'classificacao': classificacao
        },
        'recomendacoes': gerar_recomendacoes(score_final, classificacao, resultado_similaridade),
        'estatisticas_base': analisador.estatisticas_base if analisador.base_embeddings else None
    }
    
    # Salvar se path fornecido
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(relatorio, f, indent=2, ensure_ascii=False)
    
    return relatorio

def gerar_recomendacoes(score: int, classificacao: str, analise_similaridade: Dict) -> Dict:
    """
    Gera recomendações específicas baseadas no score
    """
    recomendacoes = {
        'aprovacao': '',
        'premio_sugerido': '',
        'condicoes': [],
        'dispositivos': [],
        'alertas': []
    }
    
    if classificacao == 'PREMIUM':
        recomendacoes['aprovacao'] = 'APROVAÇÃO AUTOMÁTICA'
        recomendacoes['premio_sugerido'] = 'Desconto máximo (-25%)'
        recomendacoes['condicoes'] = ['Fast-track', 'Produtos premium disponíveis']
        
    elif classificacao == 'EXCELENTE':
        recomendacoes['aprovacao'] = 'APROVAÇÃO SIMPLIFICADA'
        recomendacoes['premio_sugerido'] = 'Desconto (-15%)'
        recomendacoes['condicoes'] = ['Análise expressa']
        
    elif classificacao == 'BOM':
        recomendacoes['aprovacao'] = 'APROVAÇÃO PADRÃO'
        recomendacoes['premio_sugerido'] = 'Prêmio base'
        recomendacoes['condicoes'] = ['Processo normal']
        
    elif classificacao == 'REGULAR':
        recomendacoes['aprovacao'] = 'ANÁLISE ADICIONAL'
        recomendacoes['premio_sugerido'] = 'Majoração (+20%)'
        recomendacoes['condicoes'] = ['Vistoria prévia']
        recomendacoes['dispositivos'] = ['Rastreador recomendado']
        
    elif classificacao == 'ATENÇÃO':
        recomendacoes['aprovacao'] = 'APROVAÇÃO CONDICIONAL'
        recomendacoes['premio_sugerido'] = 'Majoração (+40%)'
        recomendacoes['condicoes'] = ['Vistoria obrigatória', 'Franquia elevada']
        recomendacoes['dispositivos'] = ['Rastreador obrigatório']
        recomendacoes['alertas'] = ['Risco elevado identificado']
        
    else:  # CRÍTICO
        recomendacoes['aprovacao'] = 'RECUSA RECOMENDADA'
        recomendacoes['premio_sugerido'] = 'Majoração (+80-100%)'
        recomendacoes['condicoes'] = ['Múltiplas restrições', 'Análise gerencial']
        recomendacoes['dispositivos'] = ['Rastreador + Bloqueador']
        recomendacoes['alertas'] = ['Risco crítico', 'Avaliar alternativas']
    
    # Adicionar insights da similaridade
    if analise_similaridade.get('taxa_sinistralidade'):
        taxa = float(analise_similaridade['taxa_sinistralidade'].strip('%')) / 100
        if taxa > 0.5:
            recomendacoes['alertas'].append(f'Alta sinistralidade em perfis similares ({taxa:.0%})')
    
    return recomendacoes

def exportar_para_ml(
    analisador: AnalisadorEmbeddings,
    dados_cliente: Dict,
    score_final: int,
    formato: str = 'numpy'
) -> np.ndarray:
    """
    Exporta dados preparados para modelos de ML
    """
    # Vetorizar cliente
    vetor_base = analisador.vetorizar_cliente(dados_cliente)
    
    # Adicionar score como feature
    vetor_completo = np.append(vetor_base, score_final)
    
    if formato == 'numpy':
        return vetor_completo
    elif formato == 'pandas':
        return pd.DataFrame([vetor_completo])
    elif formato == 'dict':
        return {f'feature_{i}': v for i, v in enumerate(vetor_completo)}
    else:
        return vetor_completo

def validar_dados_cliente(dados: Dict) -> Tuple[bool, List[str]]:
    """
    Valida se os dados do cliente estão completos
    """
    erros = []
    
    # Validações obrigatórias
    if not dados.get('localizacao', {}).get('cep'):
        erros.append('CEP é obrigatório')
    
    if not dados.get('veiculo', {}).get('marca'):
        erros.append('Marca do veículo é obrigatória')
    
    if not dados.get('veiculo', {}).get('modelo'):
        erros.append('Modelo do veículo é obrigatório')
    
    # Validações de formato
    cep = dados.get('localizacao', {}).get('cep', '')
    if cep and not cep.replace('-', '').isdigit():
        erros.append('CEP deve conter apenas números')
    
    valor_fipe = dados.get('veiculo', {}).get('valor_fipe', 0)
    if valor_fipe and valor_fipe < 0:
        erros.append('Valor FIPE não pode ser negativo')
    
    return len(erros) == 0, erros

def criar_dashboard_metricas(analisador: AnalisadorEmbeddings) -> Dict:
    """
    Cria métricas para dashboard executivo
    """
    if not analisador.estatisticas_base:
        return {}
    
    stats = analisador.estatisticas_base
    
    dashboard = {
        'metricas_principais': {
            'total_clientes': stats['total_clientes'],
            'taxa_sinistralidade': f"{stats['taxa_sinistralidade']:.1%}",
            'sinistro_medio_anual': f"{stats['media_sinistros_12m']:.2f}",
            'valor_medio_sinistro': f"R$ {stats['valor_medio_sinistro']:,.0f}"
        },
        'top_riscos': {
            'veiculos': list(stats['veiculos_mais_sinistrados'].items())[:5],
            'regioes': list(stats['regioes_criticas'].items())[:5],
            'pecas': list(stats['pecas_frequentes'].items())[:5]
        },
        'distribuicao': {
            'tipos_sinistro': stats['tipos_sinistro'],
            'percentis': {
                'p25': stats.get('percentil_25', 0),
                'p50': stats['mediana_sinistros_12m'],
                'p75': stats.get('percentil_75', 0),
                'p90': stats['percentil_90_sinistros']
            }
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return dashboard

# ================================
# CACHE E OTIMIZAÇÃO
# ================================

class CacheEmbeddings:
    """
    Sistema de cache para otimizar consultas repetidas
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def _gerar_chave(self, dados: Dict) -> str:
        """Gera chave única para o cache"""
        import hashlib
        dados_str = json.dumps(dados, sort_keys=True)
        return hashlib.md5(dados_str.encode()).hexdigest()
    
    def get(self, dados: Dict) -> Optional[Dict]:
        """Busca no cache"""
        chave = self._gerar_chave(dados)
        
        if chave in self.cache:
            entrada = self.cache[chave]
            tempo_decorrido = (datetime.now() - entrada['timestamp']).seconds
            
            if tempo_decorrido < self.ttl:
                return entrada['resultado']
            else:
                del self.cache[chave]
        
        return None
    
    def set(self, dados: Dict, resultado: Dict):
        """Salva no cache"""
        chave = self._gerar_chave(dados)
        self.cache[chave] = {
            'resultado': resultado,
            'timestamp': datetime.now()
        }
    
    def limpar(self):
        """Limpa cache expirado"""
        agora = datetime.now()
        chaves_expiradas = []
        
        for chave, entrada in self.cache.items():
            tempo_decorrido = (agora - entrada['timestamp']).seconds
            if tempo_decorrido >= self.ttl:
                chaves_expiradas.append(chave)
        
        for chave in chaves_expiradas:
            del self.cache[chave]

# Instância global do cache
cache_global = CacheEmbeddings()

# ================================
# INTEGRAÇÃO COM STREAMLIT
# ================================

def criar_interface_completa():
    """
    Interface Streamlit completa com análise dupla
    """
    st.set_page_config(
        page_title="Sistema de Score Duplo",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 Sistema de Score Duplo - APIs + Embeddings")
    
    # Upload do arquivo PKL
    uploaded_file = st.file_uploader(
        "Carregar base de embeddings (.pkl)",
        type=['pkl', 'pickle']
    )
    
    if uploaded_file:
        # Salvar temporariamente
        with open('temp_embeddings.pkl', 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Criar analisador
        analisador = AnalisadorEmbeddings('temp_embeddings.pkl')
        
        st.success(f"✅ Base carregada: {analisador.estatisticas_base['total_clientes']:,} clientes")
        
        # Mostrar dashboard
        if st.checkbox("📊 Ver Dashboard da Base"):
            dashboard = criar_dashboard_metricas(analisador)
            
            # Métricas principais
            cols = st.columns(4)
            for i, (key, value) in enumerate(dashboard['metricas_principais'].items()):
                cols[i].metric(key.replace('_', ' ').title(), value)
            
            # Top riscos
            st.subheader("⚠️ Top Riscos Identificados")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Veículos Mais Sinistrados**")
                for (marca, modelo), media in dashboard['top_riscos']['veiculos']:
                    st.write(f"• {marca} {modelo}: {media:.2f}/ano")
            
            with col2:
                st.markdown("**Regiões Críticas**")
                for cep, media in dashboard['top_riscos']['regioes']:
                    st.write(f"• CEP {cep}: {media:.2f}/ano")
            
            with col3:
                st.markdown("**Peças Mais Trocadas**")
                for peca, freq in dashboard['top_riscos']['pecas']:
                    st.write(f"• {peca}: {freq}x")

# Executar interface se chamado diretamente
if __name__ == "__main__" and 'streamlit' in globals():
    criar_interface_completa()
