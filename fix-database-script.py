#!/usr/bin/env python3
"""
Database Migration Script for SmartRig AI Tuner
Run this to fix the existing database schema issues
"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime

def fix_database():
    """Fix the existing database by adding missing columns."""
    
    # Database path
    db_path = Path.home() / ".smartrig_tuner" / "performance.db"
    
    if not db_path.exists():
        print("❌ Database not found. Please run the main application first.")
        return False
    
    print(f"📁 Found database at: {db_path}")
    
    # Backup the database first
    backup_path = db_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
    shutil.copy2(db_path, backup_path)
    print(f"✅ Created backup: {backup_path}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get current columns
        cursor.execute("PRAGMA table_info(performance_logs)")
        existing_columns = [column[1] for column in cursor.fetchall()]
        print(f"\n📋 Existing columns: {', '.join(existing_columns)}")
        
        # Define required columns
        required_columns = {
            'session_id': 'TEXT',
            'fps': 'REAL',
            'throttle_risk': 'REAL',
            'game_name': 'TEXT',
            'profile_name': 'TEXT'
        }
        
        # Add missing columns
        added_columns = []
        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE performance_logs ADD COLUMN {column_name} {column_type}")
                    added_columns.append(column_name)
                    print(f"✅ Added column: {column_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        print(f"⚠️ Warning adding {column_name}: {e}")
        
        if added_columns:
            print(f"\n🎉 Successfully added {len(added_columns)} columns: {', '.join(added_columns)}")
        else:
            print("\n✅ All required columns already exist")
        
        # Create sessions table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time DATETIME,
                end_time DATETIME,
                game_name TEXT,
                avg_fps REAL,
                avg_cpu_usage REAL,
                avg_gpu_usage REAL,
                max_temp REAL,
                profile_used TEXT
            )
        ''')
        print("✅ Sessions table verified/created")
        
        # Create indexes if they don't exist
        try:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_session ON performance_logs(session_id)')
            print("✅ Indexes verified/created")
        except:
            pass  # Indexes might already exist
        
        # Calculate throttle_risk for existing records if needed
        if 'throttle_risk' in added_columns:
            print("\n🔄 Calculating throttle_risk for existing records...")
            cursor.execute('''
                UPDATE performance_logs 
                SET throttle_risk = (
                    COALESCE(cpu_temp, 45) * 0.3 + 
                    COALESCE(gpu_temp, 45) * 0.3 + 
                    COALESCE(cpu_usage, 50) * 0.2 + 
                    COALESCE(gpu_usage, 50) * 0.2
                )
                WHERE throttle_risk IS NULL
            ''')
            updated_rows = cursor.rowcount
            print(f"✅ Updated {updated_rows} records with calculated throttle_risk")
        
        # Commit changes
        conn.commit()
        
        # Get final statistics
        cursor.execute("SELECT COUNT(*) FROM performance_logs")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("PRAGMA table_info(performance_logs)")
        final_columns = [column[1] for column in cursor.fetchall()]
        
        print(f"\n📊 Database Statistics:")
        print(f"   Total records: {total_records}")
        print(f"   Total columns: {len(final_columns)}")
        print(f"   Columns: {', '.join(final_columns)}")
        
        conn.close()
        
        print("\n✅ Database migration completed successfully!")
        print("   You can now run the main application without errors.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during migration: {e}")
        print(f"   Your backup is safe at: {backup_path}")
        return False

def verify_database():
    """Verify the database structure after migration."""
    db_path = Path.home() / ".smartrig_tuner" / "performance.db"
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check performance_logs table
        cursor.execute("PRAGMA table_info(performance_logs)")
        columns = [column[1] for column in cursor.fetchall()]
        
        required_columns = ['session_id', 'cpu_usage', 'gpu_usage', 'ram_usage', 
                          'cpu_temp', 'gpu_temp', 'cpu_freq', 'throttle_risk', 
                          'performance_score']
        
        missing = [col for col in required_columns if col not in columns]
        
        if missing:
            print(f"⚠️ Missing columns: {', '.join(missing)}")
            return False
        else:
            print("✅ All required columns present")
            return True
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SmartRig AI Tuner - Database Migration Tool")
    print("=" * 60)
    
    # Run the fix
    success = fix_database()
    
    if success:
        print("\n🔍 Verifying database structure...")
        verify_database()
    
    print("\n" + "=" * 60)
    input("Press Enter to exit...")
