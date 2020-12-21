import click

@click.group()
@click.option("--some", type=int, default=2)
@click.pass_context
def cli(ctx, some):
    ctx.ensure_object(dict)
    ctx.obj["some"] = some
    print(some)

@cli.command()
@click.option("--other", type=str, default="Hello ")
@click.pass_context
def train(ctx, other):
    """Trains a network."""
    print(other, ctx.obj["some"])

@cli.command()
@click.option("--other", type=str, default="Hello ")
def oneshot(other):
    """Oneshot pruning."""
    print(2 * other)


if __name__ == '__main__':
    cli(obj={})

