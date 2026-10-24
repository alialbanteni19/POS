import mysql.connector
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

def run_fpgrowth_analysis(min_support=0.2, min_confidence=0.5):
    # 1. Koneksi ke Database Laragon
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="dapur_umi_khalid"
        )
        print("✅ Berhasil terhubung ke database Laragon!\n")
    except Exception as e:
        print("❌ Gagal terhubung ke database:", e)
        return

    # 2. Query Ambil Data Transaksi
    query = """
    SELECT 
        t.kode_nota, 
        p.nama_produk 
    FROM transactions t
    JOIN transaction_details td ON t.id = td.transaction_id
    JOIN products p ON p.id = td.product_id;
    """
    df = pd.read_sql(query, db)
    db.close()

    print("--- 1. DATA TRANSAKSI DARI DATABASE ---")
    print(df.head(10))
    print("\n" + "="*50 + "\n")

    # 3. Format Data ke Basket Transaction List
    basket = df.groupby('kode_nota')['nama_produk'].apply(list).tolist()

    # 4. Transformasi ke Matrix One-Hot Encoding
    te = TransactionEncoder()
    te_ary = te.fit(basket).transform(basket)
    df_transformed = pd.DataFrame(te_ary, columns=te.columns_)

    # 5. Eksekusi Algoritma FP-Growth (Min Support = 20%)
    frequent_itemsets = fpgrowth(df_transformed, min_support=min_support, use_colnames=True)
    
    # 6. Formulasi Association Rules (Min Confidence = 50%)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # 7. Filter Aturan Valid (Lift Ratio > 1)
    valid_rules = rules[rules['lift'] > 1].copy()

    # Rapikan Format Tampilan
    valid_rules['antecedents'] = valid_rules['antecedents'].apply(lambda x: list(x))
    valid_rules['consequents'] = valid_rules['consequents'].apply(lambda x: list(x))

    print("--- 2. HASIL ATURAN ASOSIASI FP-GROWTH (LIFT RATIO > 1) ---")
    print(valid_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

if __name__ == "__main__":
    run_fpgrowth_analysis(min_support=0.2, min_confidence=0.5)