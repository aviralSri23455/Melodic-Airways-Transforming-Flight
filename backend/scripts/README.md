# Backend Scripts

## Community Features Setup Scripts

This folder contains scripts to set up and test community features for Aero Melody.

---

## 🚀 Quick Start

### Run the Setup (Recommended)

```bash
run_community_setup.bat
```

This will:
1. Create all database tables
2. Create database views
3. Insert sample data
4. Run tests
5. Show results

---

## 📁 Files

### 1. run_community_setup.bat
**Purpose:** One-click setup for community features
**Usage:**
```bash
run_community_setup.bat
```
**What it does:**
- Activates Python virtual environment
- Runs setup_community_db.py
- Runs test_community_features.py
- Shows success/failure report

---

### 2. setup_community_db.py
**Purpose:** Python script to execute SQL setup
**Usage:**
```bash
python setup_community_db.py
```
**What it does:**
- Reads `../sql/setup_community_features.sql`
- Executes each SQL statement
- Creates tables and views
- Inserts sample data
- Handles errors gracefully

**Output:**
```
🚀 Starting Community Features Database Setup...
============================================================
📄 Using SQL file: C:\...\backend\sql\setup_community_features.sql
Executing statement 1/50...
Executing statement 2/50...
...
✅ Community features database setup completed!
```

---

### 3. test_community_features.py
**Purpose:** Automated test suite for community features
**Usage:**
```bash
python test_community_features.py
```
**What it tests:**
- All 7 tables exist
- trending_compositions_view exists
- Sample data inserted correctly
- Queries execute without errors
- Foreign key relationships work

**Output:**
```
🧪 Community Features Test Suite
============================================================
🔍 Testing database tables...
  ✅ forum_threads: 8 rows
  ✅ forum_replies: 0 rows
  ✅ contests: 1 rows
  ✅ contest_submissions: 0 rows
  ✅ composition_likes: 0 rows
  ✅ composition_comments: 0 rows
  ✅ user_follows: 0 rows

🔍 Testing trending_compositions_view...
  ✅ View exists with X compositions

🔍 Testing sample data...
  Forum threads: 8
  Contests: 1
  ✅ Sample data looks good!

🔍 Testing community service queries...
  ✅ Trending query works! Found X recent compositions

🔍 Testing table relationships...
  ✅ Forum threads linked to users
  ✅ Contests linked to users

📊 Test Results Summary
============================================================
  ✅ PASS - Tables
  ✅ PASS - View
  ✅ PASS - Sample Data
  ✅ PASS - Queries
  ✅ PASS - Relationships

🎉 All tests passed! (5/5)
✅ Community features are working correctly!
```

---

## 🗄️ Database Objects Created

### Tables (7)
1. **forum_threads** - Discussion threads
2. **forum_replies** - Replies to threads
3. **contests** - Composition contests
4. **contest_submissions** - Contest entries
5. **composition_likes** - Like tracking
6. **composition_comments** - Comment system
7. **user_follows** - User relationships

### Views (1)
1. **trending_compositions_view** - Trending score calculation

### Sample Data
- 8 forum threads (various categories)
- 1 contest (Winter 2025 Challenge)

---

## 🔧 Troubleshooting

### Error: "venv not found"
```bash
cd ..
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Error: "Can't connect to database"
1. Check MySQL is running
2. Check `.env` file credentials
3. Verify database `melody_aero` exists

### Error: "Permission denied"
Run Command Prompt as Administrator

### Error: "Table already exists"
This is normal if you've run setup before. The script will skip existing tables.

### Tests Failing
1. Re-run `run_community_setup.bat`
2. Check MySQL user has proper permissions
3. Verify at least one user exists in database

---

## 📝 Manual Setup

If the batch file doesn't work, run manually:

### Step 1: Activate Virtual Environment
```bash
cd backend
venv\Scripts\activate
```

### Step 2: Run Setup
```bash
python scripts/setup_community_db.py
```

### Step 3: Run Tests
```bash
python scripts/test_community_features.py
```

---

## 🔍 Verification

After setup, verify in MySQL:

```sql
USE melody_aero;

-- Check tables exist
SHOW TABLES;

-- Check sample data
SELECT COUNT(*) FROM forum_threads;  -- Should be 8
SELECT COUNT(*) FROM contests;       -- Should be 1

-- Check view exists
SHOW FULL TABLES WHERE Table_type = 'VIEW';

-- Test view
SELECT * FROM trending_compositions_view LIMIT 5;
```

---

## 📚 Related Files

### SQL Schema
- `../sql/setup_community_features.sql` - Complete SQL setup

### Documentation
- `../../START_HERE.md` - Main entry point
- `../../QUICK_FIX_NOW.md` - Quick fix guide
- `../../SETUP_CHECKLIST.md` - Verification checklist
- `../../COMMUNITY_FIXES_GUIDE.md` - Comprehensive guide

---

## 🎯 What Gets Fixed

Running these scripts fixes:
- ✅ Community feed (no more dummy content)
- ✅ Contest entry (button now works)
- ✅ Database errors (tables created)
- ✅ Column errors (queries fixed)
- ✅ Forum threads (8 threads added)
- ✅ Sample contest (1 contest added)

---

## 🚀 Next Steps

After running setup:

1. **Start Backend**
   ```bash
   cd ..
   python main.py
   ```

2. **Start Frontend**
   ```bash
   cd ../..
   npm run dev
   ```

3. **Test Features**
   - Visit `http://localhost:5173/community`
   - Check forum threads display
   - Check contest displays
   - Try entering contest

---

## 📊 Success Criteria

Setup is successful when:
- ✅ All tests pass (5/5)
- ✅ No database errors
- ✅ 8 forum threads exist
- ✅ 1 contest exists
- ✅ View created successfully

---

## 🎉 All Done!

Your community features are now set up and ready to use!

For more information, see the documentation in the project root.
