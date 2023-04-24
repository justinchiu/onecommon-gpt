

class WriterMixin:
    def __init__(self, backend, refres, gen, model):
        self.gen = gen
        if gen == "sc":
            self.generate = Generate(backend.OpenAI(
                model = "text-davinci-003",
                max_tokens=512,
            ))
        elif gen == "scxy":
            self.generate = GenerateScxy(backend.OpenAI(
                model = "text-davinci-003",
                max_tokens=512,
            ))
        elif gen == "template":
            self.generate = GenerateTemplate(backend.OpenAI(
                model = "text-davinci-003",
                max_tokens=1024,
            ))
        elif gen == "templateonly":
            pass
        else:
            raise ValueError

        super(WriterMixin, self).__init__()

    def write(self):
        import pdb; pdb.set_trace()
        pass

    # GENERATION
    def generate_text(self, plan, past, view, info=None):
        if self.gen == "sc":
            return self.generate_text_sc(plan, past, view, info)
        if self.gen == "scxy":
            return self.generate_text_scxy(plan, past, view, info)
        elif self.gen == "template":
            return self.generate_text_template(plan, past, view, info)
        elif self.gen == "templateonly":
            return self.generate_text_template_only(plan, past, view, info)
        else:
            raise ValueError

    def generate_text_sc(self, plan, past, view, info=None):
        # process plan
        refs = [r["target"] for r in plan]
        size_color = process_ctx(view)
        dots = size_color[np.array(refs).any(0)]
        descs = size_color_descriptions(dots)
        descstring = []
        for size, color in descs:
            descstring.append(f"* A {size} and {color} dot")

        kwargs = dict(plan="\n".join(descstring), past="\n".join(past))
        #print("INPUT")
        #print(self.generate.print(kwargs))
        out = self.generate(kwargs)
        print("OUTPUT")
        print(out)
        return out, past + [out], None

    def generate_text_scxy(self, plan, past, view, info=None):
        # process plan
        refs = [r["target"] for r in plan]
        plan = np.array(refs).any(0)

        size_color = process_ctx(view)
        dots = size_color[plan]
        descs = size_color_descriptions(dots)
        xy = view[plan,:2]

        descstring = []
        for (size, color), (x,y) in zip(descs, xy):
            descstring.append(f"* A {size} and {color} dot (x={x:.2f},y={y:.2f})")

        kwargs = dict(plan="\n".join(descstring), past="\n".join(past))
        print("INPUT")
        print(self.generate.print(kwargs))
        out = self.generate(kwargs)
        print("OUTPUT")
        print(out)
        return out, past + [out], None

    def generate_text_template(self, plan, past, view, info=None):
        if len(plan) == 0:
            # no references...
            return "okay", past + ["okay"]

        # process plan
        refs = [r["target"] for r in plan]
        plan = np.array(refs).any(0)
        desc = render(plan, view)

        kwargs = dict(plan=desc, past="\n".join(past))
        #print("INPUT")
        #print(self.generate.print(kwargs))
        out = self.generate(kwargs)
        print("OUTPUT")
        print(out)
        return out, past + [out], None

    def generate_text_template_only(self, plan, past, view, info=None):
        if len(plan) == 0:
            # no references...
            return "okay", past + ["okay"]

        # process plan
        refs = [r["target"] for r in plan]
        plan = np.array(refs).any(0)
        desc = render(plan, view, num_buckets=3)
        print(desc)
        return desc, past + [desc], None

    def generate_new_config(self, plan, past, view, info=None):
        pass

    def generate_followup(self, plan, past, view, info=None):
        pass

    def generate_select(self, pan, past, view, info=None):
        pass
