def valida_sintaxe(s: str) -> bool:
    # Dicionário de correspondências
    correspondencias = {')': '(', '}': '{', ']': '['}
    # Pilha para armazenar os símbolos de abertura
    pilha = []

    # Percorre cada caractere da string
    for char in s:
        # Se for um símbolo de abertura, empilha
        if char in '({[':
            pilha.append(char)
        # Se for um símbolo de fechamento
        elif char in ')}]':
            # Verifica se a pilha está vazia ou se o topo não corresponde
            if not pilha or pilha[-1] != correspondencias[char]:
                return False
            # Se corresponde, desempilha
            pilha.pop()

    # No final, a pilha deve estar vazia
    return not pilha

# Exemplos de uso
print(valida_sintaxe("(){}[]"))  # True
print(valida_sintaxe("([{}])"))  # True
print(valida_sintaxe("(]"))      # False
print(valida_sintaxe("([)]"))    # False
print(valida_sintaxe("{[]}"))    # True
print(valida_sintaxe("{[{}}{}]}"))