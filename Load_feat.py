import os
import numpy as np

def feat_loader(path):
    """
    Função para carregar arquivos .feat.

    Argumentos:
    - path: caminho para o arquivo .feat ou diretório contendo vários arquivos .feat

    Retorna:
    - Array numpy com os dados do arquivo .feat ou None se nenhum arquivo for encontrado
    """
    if os.path.isfile(path):
        try:
            with open(path, 'rb') as file:
                # Ler o conteúdo do arquivo .feat
                data = file.read()

            if len(data) > 0:
                # Decodificar os dados e converter para um array numpy
                feats = np.frombuffer(data, dtype=np.float32)
                return feats
            else:
                print(f"O arquivo {path} está vazio.")
        except FileNotFoundError:
            raise FileNotFoundError(f"O arquivo {path} não foi encontrado.")
    else:
        raise ValueError(f"O caminho {path} não é um arquivo válido.")
