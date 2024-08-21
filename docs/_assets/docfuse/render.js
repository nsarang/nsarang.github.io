(function () {
    'use strict';

    // Light/Dark theme toggle
    const bodyElement = document.body;
    const observer = new MutationObserver(mutationsList => {
        // Check if any class mutations occurred
        const hasClassMutation = mutationsList.some(mutation =>
            mutation.attributeName === 'class'
        );

        if (hasClassMutation) {
            // Get all elements with the "docfuse" class
            const docfuseElements = document.querySelectorAll('.docfuse');
            const isDarkTheme = bodyElement.classList.contains('quarto-dark');

            // Iterate through the "docfuse" elements and toggle the classes
            docfuseElements.forEach(element => {
                // If the element has the "fixed-theme" class, skip it
                if (element.classList.contains('fixed-theme')) {
                    return
                }

                if (isDarkTheme) {
                    element.classList.add('dark-theme');
                    element.classList.remove('light-theme');
                } else { // light theme (default)
                    element.classList.add('light-theme');
                    element.classList.remove('dark-theme');
                }
            });
        } // End if (hasClassMutation)
    });

    observer.observe(bodyElement, { attributes: true });

    // Selection Manager
    const SelectionManager = {
        selectionStart: null,
        activeColumnType: null,

        init: function () {
            document.addEventListener('mousedown', this.handleMouseDown.bind(this));
            document.addEventListener('selectionchange', this.handleSelectionChange.bind(this));
        },

        handleMouseDown: function (event) {
            const target = event.target.closest('.docs, .code');
            if (target) {
                this.selectionStart = target;
                this.activeColumnType = target.classList.contains('docs') ? '.docs' : '.code';
                this.updateSelectability(this.activeColumnType);
            } else {
                this.resetSelection();
            }
        },

        handleSelectionChange: function () {
            if (!this.selectionStart) return;

            const selection = window.getSelection();
            if (selection.isCollapsed) return;

            this.updateSelectability(this.activeColumnType);
        },

        updateSelectability: function (columnToSelect) {
            const columnToDisable = columnToSelect === '.docs' ? '.code' : '.docs';
            this.setUserSelect(columnToSelect, 'text');
            this.setUserSelect(columnToDisable, 'none');
        },

        setUserSelect: function (selector, value) {
            document.querySelectorAll(selector).forEach(el => el.style.userSelect = value);
        },

        resetSelection: function () {
            this.selectionStart = null;
            this.activeColumnType = null;
            this.setUserSelect('.docs, .code', 'text');
        }
    };

    // Initialize the SelectionManager when the DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', SelectionManager.init.bind(SelectionManager));
    } else {
        SelectionManager.init();
    }
})();