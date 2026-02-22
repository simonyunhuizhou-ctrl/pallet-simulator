import csv
from solver import OrderSolver

order_data = []
with open(r"C:\Users\zhouy\Documents\Pallet_Simulator\Order.csv", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            name   = row.get("Item Name", "Unknown")
            qty    = int(row.get("Total Case", 0))
            length = float(row.get("Length", 0))
            width  = float(row.get("Width", 0))
            height = float(row.get("Height", 0))
            weight = float(row.get("Weight", 0))
        except ValueError:
            continue
        if qty <= 0 or length <= 0 or width <= 0 or height <= 0:
            continue
        order_data.append({
            "name": name, "qty": qty,
            "l": length, "w": width, "h": height, "weight": weight,
        })

print("Order Data Order:")
for i, d in enumerate(order_data):
    print(f"{i}: {d['name']}")

solver = OrderSolver(40.0, 48.0, 6.0, 54.0, 0.0, 4)
pallets = solver.solve(order_data)

print("\nPallets Generated:")
for i, p in enumerate(pallets):
    print(f"Pallet {i+1}: {p['skus']}")
