<template>
  <PageContent>
    <div
      :contenteditable="!readonly"
      id="originalText"
      ref="contentEditable"
      class="
        p-4
        md:min-h-screen
        focus:outline-none
        focus:ring-4
        focus:ring-offset-4
        focus:ring-offset-indigo-600
        focus:ring-white
        rounded-lg
      "
      @input="handleInput"
      @paste.prevent="onPaste"
      style="line-height: 2.5; font-size: 20px"
    />
  </PageContent>
</template>

<script>
import { onMounted, ref, watch } from "vue";
import PageContent from "./PageContent.vue";

export default {
  name: "Dashboard",
  components: { PageContent },
  props: {
    modelValue: String,
    readonly: Boolean,
  },
  setup(props, ctx) {
    const contentEditable = ref(null);
    const text = ref(props.modelValue || "");

    const handleInput = () => {
      text.value = contentEditable.value.innerText;
      ctx.emit("update:modelValue", text.value);
    };

    const onPaste = (e) => {
      const text = e.clipboardData
        ? (e.originalEvent || e).clipboardData.getData("text/plain")
        : window.clipboardData
        ? window.clipboardData.getData("Text")
        : "";

      if (document.queryCommandSupported("insertText")) {
        document.execCommand("insertText", false, text);
      } else {
        // Insert text at the current position of caret
        const range = document.getSelection().getRangeAt(0);
        range.deleteContents();

        const textNode = document.createTextNode(text);
        range.insertNode(textNode);
        range.selectNodeContents(textNode);
        range.collapse(false);

        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
      }
    };

    onMounted(() => {
      contentEditable.value.innerText = text.value;
    });

    watch(
      () => props.modelValue,
      (val) => {
        if (props.readonly) contentEditable.value.innerText = val;
      }
    );

    return {
      handleInput,
      contentEditable,
      onPaste,
    };
  },
};
</script>
