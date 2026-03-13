import os
import subprocess


def main():
    nodes_dir = os.path.join("data", "interbank", "datasets", "nodes")

    if not os.path.exists(nodes_dir):
        print(f"❌ Папка {nodes_dir} не знайдена. Перевір шляхи!")
        return

    # Шукаємо всі файли типу 2016Q1.csv і витягуємо назву кварталу
    files = [f for f in os.listdir(nodes_dir) if f.endswith(".csv")]
    quarters = sorted([f.replace(".csv", "") for f in files])

    if not quarters:
        print("❌ Ніяких CSV файлів не знайдено. Пізда.")
        return

    print(f"🔥 Знайдено {len(quarters)} кварталів: від {quarters[0]} до {quarters[-1]}. Починаємо масовий розйоб...")

    for q in quarters:
        print(f"\n{'=' * 60}")
        print(f"🚀 ЗАПУСК КВАРТАЛУ: {q}")
        print(f"{'=' * 60}\n")

        # Викликаємо main.py. Зверни увагу: БЕЗ прапорця --ablation!
        # Нам потрібні тільки фінальні порівняння B1-B6, щоб зекономити час.
        cmd = ["python", "main.py", "--quarter", q]

        try:
            # Запускаємо процес і чекаємо, поки він відпрацює
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            # Якщо один квартал впаде (наприклад, через биті дані), скрипт не зупиниться, а піде далі
            print(f"\n❌ Помилка під час обробки {q}. Йдемо до наступного...")

    print("\n✅ ВСЕ ГОТОВО! Перевіряй файл outputs/all_quarters_results.csv")


if __name__ == "__main__":
    main()