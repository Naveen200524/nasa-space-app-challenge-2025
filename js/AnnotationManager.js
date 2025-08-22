export class AnnotationManager {
    constructor() {
        this.panel = document.getElementById('annotation-panel');
        this.textArea = document.getElementById('annotation-text');
        this.saveButton = document.getElementById('save-annotation');
        this.exportButton = document.getElementById('export-annotations');
        this.closeButton = document.getElementById('close-annotations');
        this.savedContainer = document.getElementById('saved-annotations');
        
        this.annotations = [];
        this.currentEvent = null;
        
        this.setupEventListeners();
        this.loadAnnotations();
    }
    
    setupEventListeners() {
        // Save annotation
        this.saveButton?.addEventListener('click', () => {
            this.saveAnnotation();
        });
        
        // Export annotations
        this.exportButton?.addEventListener('click', () => {
            this.exportAnnotations();
        });
        
        // Close panel
        this.closeButton?.addEventListener('click', () => {
            this.hidePanel();
        });
        
        // Auto-save on text change
        this.textArea?.addEventListener('input', () => {
            this.autoSave();
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 's' && this.panel?.classList.contains('active')) {
                e.preventDefault();
                this.saveAnnotation();
            }
            
            if (e.key === 'Escape' && this.panel?.classList.contains('active')) {
                this.hidePanel();
            }
        });
    }
    
    showPanel(event = null) {
        if (!this.panel) return;
        
        this.currentEvent = event;
        this.panel.classList.add('active');
        
        if (event) {
            // Load existing annotation for this event
            const existingAnnotation = this.annotations.find(a => a.eventId === event.id);
            if (existingAnnotation && this.textArea) {
                this.textArea.value = existingAnnotation.text;
            } else if (this.textArea) {
                this.textArea.value = '';
                this.textArea.placeholder = `Add notes for ${event.planet}quake (Magnitude ${event.magnitude?.toFixed(1) || 'Unknown'})...`;
            }
        }
        
        this.updateSavedAnnotations();
        this.textArea?.focus();
    }
    
    hidePanel() {
        if (this.panel) {
            this.panel.classList.remove('active');
            this.currentEvent = null;
        }
    }
    
    saveAnnotation() {
        if (!this.textArea || !this.textArea.value.trim()) {
            this.showMessage('Please enter some text for the annotation.', 'warning');
            return;
        }
        
        const text = this.textArea.value.trim();
        const timestamp = new Date().toISOString();
        
        let annotation = {
            id: this.generateId(),
            text: text,
            timestamp: timestamp,
            eventId: this.currentEvent?.id || null,
            eventDetails: this.currentEvent ? {
                magnitude: this.currentEvent.magnitude,
                depth: this.currentEvent.depth,
                planet: this.currentEvent.planet,
                time: this.currentEvent.time
            } : null
        };
        
        // Check if annotation already exists for this event
        const existingIndex = this.annotations.findIndex(a => 
            this.currentEvent && a.eventId === this.currentEvent.id
        );
        
        if (existingIndex !== -1) {
            // Update existing annotation
            annotation.id = this.annotations[existingIndex].id;
            annotation.createdAt = this.annotations[existingIndex].createdAt || timestamp;
            annotation.updatedAt = timestamp;
            this.annotations[existingIndex] = annotation;
        } else {
            // Add new annotation
            annotation.createdAt = timestamp;
            this.annotations.unshift(annotation);
        }
        
        this.saveToStorage();
        this.updateSavedAnnotations();
        this.showMessage('Annotation saved successfully!', 'success');
        
        // Clear text area
        if (this.textArea) {
            this.textArea.value = '';
        }
    }
    
    autoSave() {
        // Implement auto-save functionality
        if (this.autoSaveTimer) {
            clearTimeout(this.autoSaveTimer);
        }
        
        this.autoSaveTimer = setTimeout(() => {
            if (this.textArea && this.textArea.value.trim() && this.currentEvent) {
                const draft = {
                    eventId: this.currentEvent.id,
                    text: this.textArea.value.trim(),
                    timestamp: new Date().toISOString(),
                    isDraft: true
                };
                
                localStorage.setItem('seismograph_annotation_draft', JSON.stringify(draft));
            }
        }, 2000);
    }
    
    updateSavedAnnotations() {
        if (!this.savedContainer) return;
        
        this.savedContainer.innerHTML = '';
        
        if (this.annotations.length === 0) {
            this.savedContainer.innerHTML = `
                <div class="no-annotations">
                    <p style="color: #9ca3af; font-style: italic; text-align: center; padding: 2rem;">
                        No annotations yet. Start by selecting a seismic event and adding your observations.
                    </p>
                </div>
            `;
            return;
        }
        
        this.annotations.forEach(annotation => {
            const annotationElement = this.createAnnotationElement(annotation);
            this.savedContainer.appendChild(annotationElement);
        });
    }
    
    createAnnotationElement(annotation) {
        const element = document.createElement('div');
        element.className = 'saved-annotation';
        element.style.cssText = `
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            position: relative;
        `;
        
        const eventInfo = annotation.eventDetails ? `
            <div style="color: #00ffff; font-size: 0.8rem; margin-bottom: 0.5rem;">
                📊 ${annotation.eventDetails.planet?.toUpperCase()} Event - 
                Magnitude ${annotation.eventDetails.magnitude?.toFixed(1) || 'Unknown'} - 
                Depth ${annotation.eventDetails.depth?.toFixed(0) || 'Unknown'}km
            </div>
        ` : '';
        
        element.innerHTML = `
            ${eventInfo}
            <div style="color: #ffffff; line-height: 1.5; margin-bottom: 0.5rem;">
                ${annotation.text}
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <small style="color: #9ca3af;">
                    ${this.formatTimestamp(annotation.timestamp)}
                    ${annotation.updatedAt ? ' (edited)' : ''}
                </small>
                <div class="annotation-actions">
                    <button class="edit-annotation" data-id="${annotation.id}" 
                            style="background: none; border: none; color: #8b5cf6; cursor: pointer; margin-right: 0.5rem;">
                        ✏️ Edit
                    </button>
                    <button class="delete-annotation" data-id="${annotation.id}"
                            style="background: none; border: none; color: #ef4444; cursor: pointer;">
                        🗑️ Delete
                    </button>
                </div>
            </div>
        `;
        
        // Add event listeners for edit and delete
        const editButton = element.querySelector('.edit-annotation');
        const deleteButton = element.querySelector('.delete-annotation');
        
        editButton?.addEventListener('click', () => {
            this.editAnnotation(annotation.id);
        });
        
        deleteButton?.addEventListener('click', () => {
            this.deleteAnnotation(annotation.id);
        });
        
        return element;
    }
    
    editAnnotation(annotationId) {
        const annotation = this.annotations.find(a => a.id === annotationId);
        if (!annotation || !this.textArea) return;
        
        this.textArea.value = annotation.text;
        this.textArea.focus();
        
        // Scroll to top to see the text area
        this.textArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    deleteAnnotation(annotationId) {
        if (!confirm('Are you sure you want to delete this annotation?')) {
            return;
        }
        
        this.annotations = this.annotations.filter(a => a.id !== annotationId);
        this.saveToStorage();
        this.updateSavedAnnotations();
        this.showMessage('Annotation deleted successfully!', 'success');
    }
    
    exportAnnotations() {
        if (this.annotations.length === 0) {
            this.showMessage('No annotations to export.', 'warning');
            return;
        }
        
        const exportData = {
            annotations: this.annotations,
            exportedAt: new Date().toISOString(),
            totalCount: this.annotations.length,
            version: '1.0',
            application: 'SeismoGuard'
        };
        
        // Create downloadable file
        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `seismograph-annotations-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
        
        this.showMessage(`Exported ${this.annotations.length} annotations successfully!`, 'success');
    }
    
    importAnnotations(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const data = JSON.parse(e.target.result);
                
                if (data.annotations && Array.isArray(data.annotations)) {
                    // Merge with existing annotations, avoiding duplicates
                    const existingIds = new Set(this.annotations.map(a => a.id));
                    const newAnnotations = data.annotations.filter(a => !existingIds.has(a.id));
                    
                    this.annotations.unshift(...newAnnotations);
                    this.saveToStorage();
                    this.updateSavedAnnotations();
                    
                    this.showMessage(`Imported ${newAnnotations.length} new annotations!`, 'success');
                } else {
                    this.showMessage('Invalid annotation file format.', 'error');
                }
            } catch (error) {
                this.showMessage('Error reading annotation file.', 'error');
                console.error('Annotation import error:', error);
            }
        };
        
        reader.readAsText(file);
    }
    
    generateId() {
        return 'annotation_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
        
        if (diffDays === 0) {
            return date.toLocaleTimeString();
        } else if (diffDays === 1) {
            return 'Yesterday ' + date.toLocaleTimeString();
        } else if (diffDays < 7) {
            return `${diffDays} days ago`;
        } else {
            return date.toLocaleDateString();
        }
    }
    
    saveToStorage() {
        try {
            localStorage.setItem('seismograph_annotations', JSON.stringify(this.annotations));
        } catch (error) {
            console.error('Failed to save annotations:', error);
            this.showMessage('Failed to save annotations locally.', 'error');
        }
    }
    
    loadAnnotations() {
        try {
            const stored = localStorage.getItem('seismograph_annotations');
            if (stored) {
                this.annotations = JSON.parse(stored);
            }
            
            // Load draft if available
            const draft = localStorage.getItem('seismograph_annotation_draft');
            if (draft && this.textArea) {
                const draftData = JSON.parse(draft);
                if (draftData.isDraft && this.currentEvent?.id === draftData.eventId) {
                    this.textArea.value = draftData.text;
                }
            }
        } catch (error) {
            console.error('Failed to load annotations:', error);
            this.annotations = [];
        }
    }
    
    getAnnotations() {
        return [...this.annotations];
    }
    
    getAnnotationsByEvent(eventId) {
        return this.annotations.filter(a => a.eventId === eventId);
    }
    
    searchAnnotations(query) {
        if (!query.trim()) return [...this.annotations];
        
        const lowercaseQuery = query.toLowerCase();
        return this.annotations.filter(annotation => 
            annotation.text.toLowerCase().includes(lowercaseQuery) ||
            (annotation.eventDetails?.planet && 
             annotation.eventDetails.planet.toLowerCase().includes(lowercaseQuery))
        );
    }
    
    showMessage(message, type = 'info') {
        // Create temporary message
        const messageElement = document.createElement('div');
        messageElement.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: ${type === 'success' ? 'rgba(16, 185, 129, 0.9)' : 
                        type === 'warning' ? 'rgba(245, 158, 11, 0.9)' : 
                        type === 'error' ? 'rgba(239, 68, 68, 0.9)' : 
                        'rgba(0, 255, 255, 0.9)'};
            color: white;
            padding: 1rem 2rem;
            border-radius: 8px;
            z-index: 10000;
            font-weight: 500;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            animation: fadeInOut 3s ease-in-out;
        `;
        
        messageElement.textContent = message;
        document.body.appendChild(messageElement);
        
        // Add animation styles
        const style = document.createElement('style');
        style.textContent = `
            @keyframes fadeInOut {
                0% { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
                20%, 80% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
                100% { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
            }
        `;
        document.head.appendChild(style);
        
        // Remove after animation
        setTimeout(() => {
            document.body.removeChild(messageElement);
            document.head.removeChild(style);
        }, 3000);
    }
}