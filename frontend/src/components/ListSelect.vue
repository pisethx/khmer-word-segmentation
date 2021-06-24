 
<template>
  <Listbox as="div" v-model="selected" class="w-96">
    <ListboxLabel class="block text-sm font-medium text-gray-300">
      {{ label }}
    </ListboxLabel>
    <div class="mt-1 relative">
      <ListboxButton
        class="
          relative
          w-full
          bg-white
          border border-gray-300
          rounded-md
          shadow-sm
          pl-3
          pr-10
          py-2
          text-left
          cursor-default
          focus:outline-none
          focus:ring-1 focus:ring-indigo-500
          focus:border-indigo-500
          sm:text-sm
        "
      >
        <span class="flex items-center">
          <span class="ml-3 block truncate">{{ selected.name }}</span>
        </span>
        <span
          class="
            ml-3
            absolute
            inset-y-0
            right-0
            flex
            items-center
            pr-2
            pointer-events-none
          "
        >
          <SelectorIcon class="h-5 w-5 text-gray-400" aria-hidden="true" />
        </span>
      </ListboxButton>

      <ListboxOptions
        class="
          absolute
          z-10
          mt-1
          w-full
          bg-white
          shadow-lg
          max-h-56
          rounded-md
          py-1
          text-base
          ring-1 ring-black ring-opacity-5
          overflow-auto
          focus:outline-none
          sm:text-sm
        "
      >
        <ListboxOption
          v-for="item in items"
          :key="item.id"
          :value="item"
          v-slot="{ active, selected }"
        >
          <li
            :class="
              'cursor-default select-none relative py-2 pl-3 pr-9 ' +
              (active ? 'text-white bg-indigo-600' : 'text-gray-900')
            "
          >
            <div class="flex items-center">
              <span
                class="ml-3 block truncate"
                :class="[
                  selected ? 'font-semibold' : 'font-normal',
                  'ml-3 block truncate',
                ]"
              >
                {{ item.name }}
              </span>
            </div>

            <span
              v-if="selected"
              :class="[
                active ? 'text-white' : 'text-indigo-600',
                'absolute inset-y-0 right-0 flex items-center pr-4',
              ]"
            >
              <CheckIcon class="h-5 w-5" aria-hidden="true" />
            </span>
          </li>
        </ListboxOption>
      </ListboxOptions>
    </div>
  </Listbox>
</template>

<script>
import { ref, watch } from "vue";
import {
  Listbox,
  ListboxButton,
  ListboxLabel,
  ListboxOption,
  ListboxOptions,
} from "@headlessui/vue";
import { CheckIcon, SelectorIcon } from "@heroicons/vue/solid";

export default {
  props: {
    label: String,
    items: Array,
    modelValue: [String, Number],
  },
  components: {
    Listbox,
    ListboxButton,
    ListboxLabel,
    ListboxOption,
    ListboxOptions,
    CheckIcon,
    SelectorIcon,
  },
  setup(props, ctx) {
    const items = ref(props.items);
    const selected = ref(
      items.value.find((item) => item.id === props.modelValue)
    );

    watch(
      () => selected.value,
      (val) => ctx.emit("update:modelValue", val.id)
    );

    return {
      items,
      selected,
    };
  },
};
</script>
