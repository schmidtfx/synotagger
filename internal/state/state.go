package state

import (
	"database/sql"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

type Record struct {
	Path        string
	Size        int64
	MtimeNs     int64
	SHA256      string
	LastScanned int64
	NSFWScore   float32
	TagsWritten bool
	Err         string
}

type DB struct {
	sql *sql.DB
}

func Open(path string) (*DB, error) {
	db, err := sql.Open("sqlite3", path)
	if err != nil { return nil, err }
	if _, err := db.Exec(`
		PRAGMA journal_mode=WAL;
		PRAGMA synchronous=NORMAL;
		CREATE TABLE IF NOT EXISTS files(
			path TEXT PRIMARY KEY,
			size INTEGER NOT NULL,
			mtime_ns INTEGER NOT NULL,
			sha256 TEXT,
			last_scanned INTEGER NOT NULL,
			nsfw_score REAL NOT NULL,
			tags_written INTEGER NOT NULL,
			err TEXT
		);
		CREATE INDEX IF NOT EXISTS idx_files_sha ON files(sha256);
	`); err != nil {
		db.Close()
		return nil, err
	}
	return &DB{sql: db}, nil
}

func (d *DB) Close() error { return d.sql.Close() }

// Get retrieves a record by path.
func (d *DB) Get(path string) (*Record, error) {
	row := d.sql.QueryRow(`SELECT path,size,mtime_ns,sha256,last_scanned,nsfw_score,tags_written,err FROM files WHERE path=?`, path)
	var rec Record
	var tags int
	if err := row.Scan(&rec.Path, &rec.Size, &rec.MtimeNs, &rec.SHA256, &rec.LastScanned, &rec.NSFWScore, &tags, &rec.Err); err != nil {
		if err == sql.ErrNoRows { return nil, nil }
		return nil, err
	}
	rec.TagsWritten = tags == 1
	return &rec, nil
}

// FindByHash returns a record with the same content hash (used to detect renames).
func (d *DB) FindByHash(hash string) (*Record, error) {
	row := d.sql.QueryRow(`SELECT path,size,mtime_ns,sha256,last_scanned,nsfw_score,tags_written,err FROM files WHERE sha256=?`, hash)
	var rec Record
	var tags int
	if err := row.Scan(&rec.Path, &rec.Size, &rec.MtimeNs, &rec.SHA256, &rec.LastScanned, &rec.NSFWScore, &tags, &rec.Err); err != nil {
		if err == sql.ErrNoRows { return nil, nil }
		return nil, err
	}
	rec.TagsWritten = tags == 1
	return &rec, nil
}

// Upsert inserts or updates a record.
func (d *DB) Upsert(r *Record) error {
	_, err := d.sql.Exec(`
		INSERT INTO files(path,size,mtime_ns,sha256,last_scanned,nsfw_score,tags_written,err)
		VALUES(?,?,?,?,?,?,?,?)
		ON CONFLICT(path) DO UPDATE SET
			size=excluded.size,
			mtime_ns=excluded.mtime_ns,
			sha256=excluded.sha256,
			last_scanned=excluded.last_scanned,
			nsfw_score=excluded.nsfw_score,
			tags_written=excluded.tags_written,
			err=excluded.err
	`, r.Path, r.Size, r.MtimeNs, r.SHA256, r.LastScanned, r.NSFWScore, boolToInt(r.TagsWritten), r.Err)
	return err
}

func boolToInt(b bool) int { if b { return 1 }; return 0 }

func NowUnix() int64 { return time.Now().Unix() }
