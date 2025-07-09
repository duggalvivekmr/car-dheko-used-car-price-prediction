with open("variant.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

quoted = [f'"{line}"' for line in lines]
output = ", ".join(quoted)

with open("quoted_variants.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("✅ Quoted variant names saved to quoted_variants.txt")
