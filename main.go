package main

import (
	"encoding/json"
	"fmt"
	"html/template"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

type MCQ struct {
	ID       int      `json:"id"`
	Question string   `json:"question"`
	Options  []string `json:"options"`
	Topic    string   `json:"topic"`
}

type PageData struct {
	Subject string
	Page    string
	MCQs    []MCQ
}

func main() {
	mux := http.NewServeMux()

	// Route to view a specific page: /view/maths/page_8
	mux.HandleFunc("/view/", func(w http.ResponseWriter, r *http.Request) {
		// Trim prefix and split: "maths/page_8" -> ["maths", "page_8"]
		relPath := strings.TrimPrefix(r.URL.Path, "/view/")
		parts := strings.Split(relPath, "/")

		if len(parts) < 2 {
			http.Error(w, "Invalid path format. Use /view/subject/page_n", 400)
			return
		}

		subject := parts[0]
		pageName := parts[1]
		fullPath := filepath.Join("data", subject, pageName+".json")

		file, err := os.ReadFile(fullPath)
		if err != nil {
			http.Error(w, "JSON file not found: "+fullPath, 404)
			return
		}

		var wrapper struct {
			Questions []MCQ `json:"questions"`
		}
		if err := json.Unmarshal(file, &wrapper); err != nil {
			http.Error(w, "Error parsing JSON", 500)
			return
		}
		mcqs := wrapper.Questions

		tmpl, err := template.ParseFiles("templates/index.html")
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}

		// Execute only the "questions" define block for HTMX
		tmpl.ExecuteTemplate(w, "questions", PageData{
			Subject: subject,
			Page:    pageName,
			MCQs:    mcqs,
		})
	})

	// Main Index
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		tmpl := template.Must(template.ParseFiles("templates/index.html"))
		tmpl.Execute(w, nil)
	})

	fmt.Println("🚀 Server starting at http://localhost:8080")
	http.ListenAndServe(":8083", mux)
}