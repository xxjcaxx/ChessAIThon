export { initStyle, initTemplate };

const initStyle = style => {
    const styleElement = document.createElement("style");
    styleElement.textContent = style;
    return styleElement;
}

const initTemplate = (template) => {
    const templateWrapper = document.createElement("div");
    templateWrapper.innerHTML = template;
    return templateWrapper.firstElementChild;
}
