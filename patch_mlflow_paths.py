import sqlite3
import os

def patch_db():
    db_path = 'mlflow.db'
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return
    
    print("Patching MLflow SQLite paths for Docker...")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Update runs table
    c.execute("SELECT run_uuid, artifact_uri FROM runs")
    runs = c.fetchall()
    for run_uuid, uri in runs:
        if uri and '/mlruns/' in uri.replace('\\', '/'):
            new_uri = 'file:///app/mlruns/' + uri.replace('\\', '/').split('/mlruns/')[-1]
            c.execute("UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?", (new_uri, run_uuid))
            print(f"Patched run {run_uuid}")
            
    # Update model_versions table
    c.execute("SELECT name, version, source FROM model_versions")
    versions = c.fetchall()
    for name, version, source in versions:
        if source and '/mlruns/' in source.replace('\\', '/'):
            new_source = 'file:///app/mlruns/' + source.replace('\\', '/').split('/mlruns/')[-1]
            c.execute("UPDATE model_versions SET source = ? WHERE name = ? AND version = ?", (new_source, name, version))
            print(f"Patched model_version {name} v{version}")
            
    conn.commit()
    conn.close()
    print("MLflow SQLite patch complete.")

if __name__ == '__main__':
    patch_db()
